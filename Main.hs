{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}

module Main where

import qualified Graphics.Gloss as G
import qualified Graphics.Gloss.Interface.Pure.Game as G
import qualified Data.Massiv.Array as M
import Data.Massiv.Array (Array, Comp(..), S, D, DI, Ix2(..), Ix3(..), (!), Sz(..))
import qualified Data.Massiv.Array.Numeric as M
import Control.Monad (forM_, replicateM)
import System.Random (randomRIO)
import qualified Data.Vector.Unboxed as V
import qualified Data.ByteString.Lazy as BS
import Data.Binary
import Data.Binary.Get
import System.IO.Error (catchIOError)
import Debug.Trace (trace)

-- Neural Network Types
type Weight = Float
type Bias = Float
type Layer = Array S Ix2 Float

data Network = Network 
    { conv1Weights :: [Array S Ix2 Weight]
    , conv1Biases :: [Bias]
    , conv2Weights :: [Array S Ix3 Weight]  -- 3D kernels for second convolution
    , conv2Biases :: [Bias]
    , fcWeights :: Array S Ix2 Weight
    , fcBiases :: [Bias]
    , outputWeights :: Array S Ix2 Weight
    , outputBiases :: [Bias]
    }

data World = World
    { inputImage :: Array S Ix2 Float
    , conv1Features :: [Array S Ix2 Float]
    , pool1Features :: [Array S Ix2 Float]
    , conv2Features :: [Array S Ix2 Float]
    , pool2Features :: [Array S Ix2 Float]
    , fcFeatures :: [Float]
    , outputFeatures :: [Float]
    , network :: Network
    , currentStage :: Stage
    , mousePos :: (Float, Float)
    , isDrawing :: Bool
    , shouldProcess :: Bool
    , stageIndex :: Int
    , stages :: [Stage]  -- Added stages to World
    , stageDescriptions :: [String]
    }

data Stage = Drawing | Conv1 | Pool1 | Conv2 | Pool2 | FullyConnected | Prediction
    deriving (Eq, Show)

-- Neural Network Operations
relu :: Float -> Float
relu x = max 0 x

convolve :: Array S Ix2 Float -> Array S Ix2 Float -> Array S Ix2 Float
convolve input kernel =
    let M.Sz (Ix2 h w) = M.size input
        M.Sz (Ix2 kh kw) = M.size kernel
        padH = kh `div` 2
        padW = kw `div` 2
        getPaddedPixel i j =
            if i >= 0 && i < h && j >= 0 && j < w
            then input ! Ix2 i j
            else 0.0
        _ = trace ("Kernel size: " ++ show (kh, kw) ++ ", max kernel val: " ++ show (M.foldlS max 0 kernel)) $ ()
    in M.makeArray M.Seq (Sz (Ix2 h w)) $ \(Ix2 i j) ->
        let result = sum [ (getPaddedPixel (i + ki - padH) (j + kj - padW)) * (kernel ! Ix2 ki kj)
                        | ki <- [0..kh-1]
                        , kj <- [0..kw-1]
                        ]
        in result

convolve3D :: Array S Ix3 Float -> Array S Ix3 Float -> Array S Ix2 Float
convolve3D input kernel =
    let M.Sz (M.Ix3 d h w) = M.size input
        M.Sz (M.Ix3 dk kh kw) = M.size kernel
        padH = kh `div` 2
        padW = kw `div` 2
        get_pixel c i j = if i >= 0 && i < h && j >= 0 && j < w
                         then input ! M.Ix3 c i j
                         else 0
        get_kernel c i j = kernel ! M.Ix3 c i j
    in M.makeArray M.Seq (Sz (Ix2 h w)) $ \(Ix2 i j) ->
        sum [ get_pixel c (i + ki - padH) (j + kj - padW) * get_kernel c ki kj
            | c <- [0..d-1]
            , ki <- [0..kh-1]
            , kj <- [0..kw-1]
            ]

convLayer :: [Array S Ix2 Float] -> [Bias] -> Array S Ix2 Float -> [Array S Ix2 Float]
convLayer weights biases input = 
    let maxInput = M.foldlS max 0 input
        _ = trace ("Conv layer input max: " ++ show maxInput) ()
        results = zipWith (\(w, i) b -> 
            let maxWeight = M.foldlS max 0 w
                _ = trace ("Conv filter " ++ show i ++ " max weight: " ++ show maxWeight) ()
                conv = convolve input w
                maxConv = M.foldlS max 0 conv
                withBias = M.computeS $ M.map (relu . (+b)) conv
                maxResult = M.foldlS max 0 withBias
                _ = trace ("Conv " ++ show i ++ " - before bias max: " ++ show maxConv ++ 
                          ", after bias+relu max: " ++ show maxResult) ()
            in withBias) (zip weights [0..]) biases
    in results

convLayer3D :: [Array S Ix3 Float] -> [Bias] -> Array S Ix3 Float -> [Array S Ix2 Float]
convLayer3D weights biases input = 
    zipWith (\w b -> 
        let conv = convolve3D input w  -- Already returns Array S Ix2 Float
            result = M.computeS $ M.map (relu . (+b)) conv
        in result) weights biases

maxPool :: Array S Ix2 Float -> Array S Ix2 Float
maxPool arr =
    let M.Sz (Ix2 h w) = M.size arr
        ph = h `div` 2
        pw = w `div` 2
        result = M.makeArray M.Seq (Sz (Ix2 ph pw)) $ \(Ix2 i j) ->
            if 2*i + 1 < h && 2*j + 1 < w
            then maximum [ arr ! Ix2 (2*i + di) (2*j + dj) | di <- [0,1], dj <- [0,1] ]
            else 0
        maxPoolVal = M.foldlS max 0 result
        _ = trace ("Pool max value: " ++ show maxPoolVal) $ ()
    in result

fullyConnected :: Array S Ix2 Weight -> [Bias] -> [Float] -> [Float]
fullyConnected weights biases inputs = 
    let flatWeights = M.toList weights
        numInputs = length inputs
        layerSize = length biases
        maxWeight = maximum flatWeights
        maxInput = maximum inputs
        _ = trace ("FC layer - inputs: " ++ show (take 5 inputs) ++ "...") $ ()
        computeNeuron n = 
            let weights = take numInputs $ drop (n * numInputs) flatWeights
                dotProduct = sum (zipWith (*) weights inputs)
                result = dotProduct + biases !! n  -- Remove ReLU from final layers
                _ = if n < 3 then
                    trace ("Neuron " ++ show n ++ 
                          " weights: " ++ show (take 5 weights) ++ 
                          " dot product: " ++ show dotProduct ++ 
                          ", bias: " ++ show (biases !! n) ++
                          ", result: " ++ show result) $ ()
                    else ()
            in result
    in map computeNeuron [0..layerSize-1]

softmax :: [Float] -> [Float]
softmax xs = 
    let maxVal = maximum xs
        shiftedXs = map (\x -> x - maxVal) xs
        expXs = map exp shiftedXs
        sumExp = sum expXs
        result = map (/sumExp) expXs
        _ = trace ("Softmax - raw input: " ++ show xs ++ 
                  "\nShifted input: " ++ show shiftedXs ++
                  "\nExp values: " ++ show expXs ++
                  "\nSum exp: " ++ show sumExp ++
                  "\nFinal output: " ++ show result) $ ()
    in result

forwardPass :: Network -> Array S Ix2 Float -> ([Array S Ix2 Float], [Array S Ix2 Float], [Array S Ix2 Float], [Array S Ix2 Float], [Float], [Float])
forwardPass Network{..} input = 
    let !inputMax = M.foldlS max 0 input
        _ = trace ("Input max value: " ++ show inputMax) ()
        
        -- Ensure input has values
        !validInput = if inputMax <= 0 
                     then trace "WARNING: Input is all zeros!" input
                     else input

        -- Process with forced evaluation
        !c1 = map (\(!w, !b) -> 
                let !result = M.computeS $ M.map (relu . (+b)) $ convolve validInput w
                    !maxVal = M.foldlS max 0 result
                    _ = trace ("Conv1 output max: " ++ show maxVal) ()
                in result) 
            (zip conv1Weights conv1Biases)
        !c1MaxVal = maximum $ map (M.foldlS max 0) c1
        _ = trace ("CONV1 CHECK: Max conv1 value: " ++ show c1MaxVal) $ ()

        !p1 = map maxPool c1
        !p1MaxVal = maximum $ map (M.foldlS max 0) p1
        _ = trace ("POOL1 CHECK: Max pool1 value: " ++ show p1MaxVal) $ ()

        !inputToConv2 = M.computeS $ (M.makeArray M.Seq (Sz (M.Ix3 (length p1) h w)) 
                       (\(M.Ix3 c y x) -> p1 !! c ! Ix2 y x) :: Array D Ix3 Float)
            where Sz (Ix2 h w) = M.size (head p1)

        -- Force evaluate conv2
        c2Raw = convLayer3D conv2Weights conv2Biases inputToConv2
        !c2 = map (M.computeS . M.delay) c2Raw
        !c2MaxVal = maximum $ map (M.foldlS max 0) c2
        _ = trace ("CONV2 CHECK: Max conv2 value: " ++ show c2MaxVal) $ ()

        !p2 = map maxPool c2
        !p2MaxVal = maximum $ map (M.foldlS max 0) p2
        _ = trace ("POOL2 CHECK: Max pool2 value: " ++ show p2MaxVal) $ ()

        !flattenedP2 = concat [M.toList feature | feature <- p2]
        !flattenedMaxVal = maximum flattenedP2
        _ = trace ("FLATTEN CHECK: Max flattened value: " ++ show flattenedMaxVal) $ ()

        -- First FC layer
        !fc = fullyConnected fcWeights fcBiases flattenedP2
        !fcMaxVal = maximum fc
        _ = trace ("FC layer values: " ++ show fc) $ ()

        -- Output layer (no ReLU)
        !preOutput = fullyConnected outputWeights outputBiases fc
        _ = trace ("Pre-softmax values: " ++ show preOutput) $ ()
        
        !output = softmax preOutput
        _ = trace ("Post-softmax values: " ++ show output) $ ()

        -- Force evaluation of key values
        !_ = c1MaxVal `seq` p1MaxVal `seq` c2MaxVal `seq` p2MaxVal `seq` 
             fcMaxVal `seq` flattenedMaxVal `seq` ()
    in (c1, p1, c2, p2, fc, output)

drawAtPoint :: (Float, Float) -> Array S Ix2 Float -> Array S Ix2 Float
drawAtPoint (x, y) arr =
    let (cx, cy) = screenToGrid (x, y)
        newImage = M.makeArray M.Seq (M.size arr) $ \(Ix2 i j) ->
            let oldVal = arr ! Ix2 i j
                dist = sqrt ((fromIntegral i - cx)^2 + (fromIntegral j - cy)^2)
                !brushValue = if dist < 3 then 1.0 else 0.0  -- Simplified brush
                !newVal = max oldVal brushValue
                !_ = if brushValue > 0 
                     then trace ("Set pixel " ++ show (i,j) ++ " to " ++ show newVal) ()
                     else ()
            in newVal
        !maxNewVal = M.foldlS max 0 newImage
        !_ = trace ("Max value in image after drawing: " ++ show maxNewVal) ()
    in if maxNewVal > 0 then newImage else arr
  where
    screenToGrid :: (Float, Float) -> (Float, Float)
    screenToGrid (sx, sy) =
        let gridX = (sx + 400) * 28 / 800
            gridY = (sy + 300) * 28 / 600
            _ = trace ("Converting screen " ++ show (sx, sy) ++ " to grid " ++ show (gridX, gridY)) $ ()
        in (gridX, gridY)

renderDrawingStage :: Array S Ix2 Float -> G.Picture
renderDrawingStage img = 
    G.translate (-280.0) (-280.0) $
    G.scale 20.0 20.0 $ G.pictures 
        [G.translate (fromIntegral x) (fromIntegral y) $ 
         G.color (G.makeColor 0 0 0 (img ! Ix2 y x)) $
         G.rectangleSolid 1 1
        | x <- [0..27], y <- [0..27]]

renderFeatureMaps :: [Array S Ix2 Float] -> G.Picture
renderFeatureMaps features = 
    let n = length features
        cols = ceiling (sqrt (fromIntegral n :: Double)) :: Int
        rows = ceiling ((fromIntegral n :: Double) / fromIntegral cols) :: Int
        scale = min (600.0 / fromIntegral rows) (800.0 / fromIntegral cols)
        positions = [(fromIntegral x, fromIntegral y) | y <- [0..rows-1], x <- [0..cols-1]]
        
        -- Get dimensions of the feature maps
        Sz (Ix2 h w) = M.size (head features)
        
        -- Adjust renderFeatureMap for each feature's dimensions
        renderFeatureMap feature =
            G.pictures 
                [G.translate (fromIntegral x) (fromIntegral y) $
                 G.color (G.makeColor v v v 1) $
                 G.rectangleSolid 1 1
                | x <- [0..w-1], y <- [0..h-1]
                , let v = min 1.0 $ max 0.0 $ feature ! Ix2 y x]
                
    in G.pictures 
        [G.translate (x * scale - 400.0) (y * scale - 300.0) $
         G.scale (scale/fromIntegral w) (scale/fromIntegral h) $
         renderFeatureMap f
        | (f, (x, y)) <- zip features positions]

renderFeatureMap :: Array S Ix2 Float -> G.Picture
renderFeatureMap feature =
    G.pictures 
        [G.translate (fromIntegral x) (fromIntegral y) $
         G.color (G.makeColor v v v 1) $
         G.rectangleSolid 1 1
        | x <- [0..27], y <- [0..27]
        , let v = min 1.0 $ max 0.0 $ feature ! Ix2 y x]

renderFullyConnected :: [Float] -> G.Picture
renderFullyConnected activations =
    let n = length activations
        width = 800.0
        height = 600.0
        spacing = width / fromIntegral n
    in G.pictures 
        [G.translate (x * spacing - width/2.0) 0 $
         G.color (G.makeColor v v v 1) $
         G.circleSolid 10.0
        | (v, x) <- zip activations [0.0, spacing .. width - spacing]]

renderPrediction :: [Float] -> G.Picture
renderPrediction probs =
    let barWidth = 60.0
        spacing = 20.0
        totalWidth = (barWidth + spacing) * 10.0
        maxHeight = 500.0
        _ = trace ("Rendering predictions: " ++ show probs) $ ()
    in G.translate (-totalWidth/2.0) (-250.0) $ G.pictures
        [G.translate (fromIntegral i * (barWidth + spacing)) 0 $
         G.pictures
            [G.color G.blue $
             G.rectangleSolid barWidth (p * maxHeight),
             G.translate 0 (-30.0) $
             G.scale 0.1 0.1 $
             G.text (show i ++ ": " ++ show (p * 100) ++ "%")]
        | (p, i) <- zip probs [0..9]]

handleEvent :: G.Event -> World -> World
handleEvent (G.EventKey (G.SpecialKey G.KeyEnter) G.Down _ _) world@World{..} =
    let nextIndex = (stageIndex + 1) `mod` length stages
    in world { stageIndex = nextIndex, currentStage = stages !! nextIndex, shouldProcess = True }
handleEvent (G.EventKey (G.MouseButton G.LeftButton) G.Down _ pos) world =
    world { isDrawing = True, mousePos = pos }
handleEvent (G.EventKey (G.MouseButton G.LeftButton) G.Up _ _) world =
    world { isDrawing = False }
handleEvent (G.EventMotion pos) world@World{..} =
    if isDrawing
    then world { mousePos = pos }
    else world
handleEvent _ world = world

update :: Float -> World -> World
update _ world@World{..}
    | shouldProcess = 
        let _ = putStrLn "\n=== Starting Forward Pass ==="  -- Use putStrLn instead of trace
            (!c1, !p1, !c2, !p2, !fc, !out) = forwardPass network inputImage
            -- Calculate and USE the values to force evaluation
            !maxInput = M.foldlS max 0 inputImage
            !maxC1 = maximum $ map (M.foldlS max 0) c1
            !maxP1 = maximum $ map (M.foldlS max 0) p1
            !maxC2 = maximum $ map (M.foldlS max 0) c2
            !maxP2 = maximum $ map (M.foldlS max 0) p2
            !maxFC = maximum fc
            !maxOut = maximum out

            -- Use all the values in the output features
            !outputFeatures' = zipWith (\x m -> 
                                let !_ = trace ("Stage max: " ++ show m) ()
                                in x) 
                             out [maxInput, maxC1, maxP1, maxC2, maxP2, maxFC, maxOut]

            result = world { conv1Features = c1
                         , pool1Features = p1
                         , conv2Features = c2
                         , pool2Features = p2
                         , fcFeatures = fc
                         , outputFeatures = outputFeatures'
                         , shouldProcess = False
                         }
        in result `seq` maxInput `seq` maxC1 `seq` maxP1 `seq` 
           maxC2 `seq` maxP2 `seq` maxFC `seq` maxOut `seq` result

    | isDrawing = 
        let !newImage = drawAtPoint mousePos inputImage
            !maxVal = M.foldlS max 0 newImage
            -- Force maxVal evaluation
            !resultImage = if maxVal > 0 
                         then trace ("Max image value: " ++ show maxVal) newImage
                         else newImage
        in world { inputImage = resultImage }
    | otherwise = world

render :: World -> G.Picture
render World{..} = case currentStage of
    Drawing -> renderDrawingStage inputImage
    Conv1 -> renderFeatureMaps conv1Features
    Pool1 -> renderFeatureMaps pool1Features
    Conv2 -> renderFeatureMaps conv2Features
    Pool2 -> renderFeatureMaps pool2Features
    FullyConnected -> renderFullyConnected fcFeatures
    Prediction -> renderPrediction outputFeatures

getConvKernel :: Int -> Int -> Get (Array S Ix2 Float)
getConvKernel h w = do
    values <- replicateM (h * w) getFloatle
    return $ M.makeArray M.Seq (Sz (Ix2 h w)) $ \(Ix2 i j) ->
        values !! (i * w + j)

getConvKernel3D :: Int -> Int -> Int -> Get (Array S Ix3 Float)
getConvKernel3D d h w = do
    values <- replicateM (d * h * w) getFloatle
    return $ M.makeArray M.Seq (Sz (M.Ix3 d h w)) $ \(M.Ix3 c i j) ->
        values !! (c * h * w + i * w + j)

getArray2D :: Int -> Int -> Get (Array S Ix2 Float)
getArray2D h w = do
    values <- replicateM (h * w) getFloatle
    return $ M.makeArray M.Seq (Sz (Ix2 h w)) $ \(Ix2 i j) ->
        values !! (i * w + j)

loadNetwork :: FilePath -> IO (Either String Network)
loadNetwork path = catchIOError
    (do
        content <- BS.readFile path
        return $ decodeWeights content
    )
    (\e -> return $ Left $ "Error reading file: " ++ show e)

decodeWeights :: BS.ByteString -> Either String Network
decodeWeights bs = 
    case runGetOrFail getNetwork bs of
        Left (_, _, err) -> Left $ "Error decoding weights: " ++ err
        Right (_, _, network) -> 
            let firstConv = head (conv1Weights network)
                firstConvBias = head (conv1Biases network)
                -- Print first 2x2 of first conv kernel
                _ = trace ("First conv kernel 2x2: " ++ 
                          show [ firstConv ! Ix2 i j 
                              | i <- [0..1], j <- [0..1]]) $ ()
                _ = trace ("First conv bias: " ++ show firstConvBias) $ ()
                -- Print first few FC weights
                firstFCWeights = take 5 $ M.toList (fcWeights network)
                firstFCBias = head (fcBiases network)
                _ = trace ("First FC weights: " ++ show firstFCWeights) $ ()
                _ = trace ("First FC bias: " ++ show firstFCBias) $ ()
            in Right network
  where
    getNetwork = do
        conv1Weights <- replicateM 6 $ getConvKernel 5 5
        _ <- return $ trace ("Conv1 first kernel values: " ++ show (take 4 $ M.toList (head conv1Weights))) ()
        conv1Biases <- replicateM 6 getFloatle
        _ <- return $ trace ("Conv1 first bias: " ++ show (head conv1Biases)) ()
        
        conv2Weights <- replicateM 16 $ getConvKernel3D 6 5 5
        conv2Biases <- replicateM 16 getFloatle
        
        fcWeights <- getArray2D 120 84
        _ <- return $ trace ("FC first weights: " ++ show (take 5 $ M.toList fcWeights)) ()
        fcBiases <- replicateM 84 getFloatle
        _ <- return $ trace ("FC first bias: " ++ show (head fcBiases)) ()
        
        outputWeights <- getArray2D 84 10
        outputBiases <- replicateM 10 getFloatle
        return Network{..}

main :: IO ()
main = do
    putStrLn "Starting program..."
    netResult <- loadNetwork "mnist_weights.bin"
    case netResult of
        Left err -> putStrLn $ "Failed to load network: " ++ err
        Right net -> do
            let !conv1FirstKernel = head (conv1Weights net)
                !conv1FirstBias = head (conv1Biases net)
                !fcFirstWeight = M.foldlS max 0 (fcWeights net)
            putStrLn $ "First conv kernel sample: " ++ show (conv1FirstKernel ! Ix2 0 0)
            putStrLn $ "First conv bias: " ++ show conv1FirstBias
            putStrLn $ "Max FC weight: " ++ show fcFirstWeight
            putStrLn $ "FC Weights size: " ++ show (M.size (fcWeights net))
            putStrLn $ "Output Weights size: " ++ show (M.size (outputWeights net))
            putStrLn "Network loaded successfully"
            let initialImage = M.makeArray M.Seq (Sz (Ix2 28 28)) (const 0) :: Array S Ix2 Float
                stages = [Drawing, Conv1, Pool1, Conv2, Pool2, FullyConnected, Prediction]
                world = World {
                    inputImage = initialImage,
                    conv1Features = [],
                    pool1Features = [],
                    conv2Features = [],
                    pool2Features = [],
                    fcFeatures = [],
                    outputFeatures = replicate 10 0.1,
                    network = net,
                    currentStage = Drawing,
                    mousePos = (0, 0),
                    isDrawing = False,
                    shouldProcess = False,
                    stageIndex = 0,
                    stages = stages,
                    stageDescriptions = ["Drawing", "Conv1", "Pool1", "Conv2", "Pool2", "FullyConnected", "Prediction"]
                }
            putStrLn "Starting visualization..."
            G.play
                (G.InWindow "CNN Visualization" (800, 600) (100, 100))
                G.white
                30
                world
                render
                handleEvent
                update