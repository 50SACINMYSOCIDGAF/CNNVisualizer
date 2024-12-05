{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}

module Main where

import qualified Graphics.Gloss as G
import qualified Graphics.Gloss.Interface.Pure.Game as G
import qualified Data.Massiv.Array as M
import Data.Massiv.Array (Array, Comp(..), S, Ix2(..), Ix3(..), (!), Sz(..))
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
        outputH = h - kh + 1
        outputW = w - kw + 1
    in M.makeArray M.Seq (Sz (Ix2 outputH outputW)) $ \(Ix2 i j) ->
        sum [ (input ! Ix2 (i + ki) (j + kj)) * (kernel ! Ix2 ki kj)
            | ki <- [0..kh-1]
            , kj <- [0..kw-1]
            ]

convolve3D :: Array S Ix3 Float -> Array S Ix3 Float -> Array S Ix2 Float
convolve3D input kernel =
    let M.Sz (M.Ix3 d h w) = M.size input
        M.Sz (M.Ix3 dk kh kw) = M.size kernel
        outputH = h - kh + 1
        outputW = w - kw + 1
        get_pixel c i j = if i >= 0 && i < h && j >= 0 && j < w
                         then input ! M.Ix3 c i j
                         else 0
        get_kernel c i j = kernel ! M.Ix3 c i j
    in M.makeArray M.Seq (Sz (Ix2 outputH outputW)) $ \(Ix2 i j) ->
        sum [ get_pixel c (i + ki) (j + kj) * get_kernel c ki kj
            | c <- [0..d-1]
            , ki <- [0..kh-1]
            , kj <- [0..kw-1]
            ]

convLayer :: [Array S Ix2 Float] -> [Bias] -> Array S Ix2 Float -> [Array S Ix2 Float]
convLayer weights biases input = 
    zipWith (\w b -> M.computeS $ M.map (relu . (+b)) $ convolve input w) weights biases

convLayer3D :: [Array S Ix3 Float] -> [Bias] -> Array S Ix3 Float -> [Array S Ix2 Float]
convLayer3D weights biases input = 
    zipWith (\w b -> M.computeS $ M.map (relu . (+b)) $ convolve3D input w) weights biases

maxPool :: Array S Ix2 Float -> Array S Ix2 Float
maxPool arr =
    let M.Sz (Ix2 h w) = M.size arr
        ph = h `div` 2
        pw = w `div` 2
    in M.makeArray M.Seq (Sz (Ix2 ph pw)) $ \(Ix2 i j) ->
        if 2*i + 1 < h && 2*j + 1 < w
        then maximum [ arr ! Ix2 (2*i + di) (2*j + dj) | di <- [0,1], dj <- [0,1] ]
        else 0  -- or some other default value

fullyConnected :: Array S Ix2 Weight -> [Bias] -> [Float] -> [Float]
fullyConnected weights biases inputs = 
    let flatWeights = M.toList weights
        layerSize = length biases
        computeNeuron n = 
            relu $ sum (zipWith (*) inputs (take (length inputs) $ drop (n * length inputs) flatWeights)) + biases !! n
    in map computeNeuron [0..layerSize-1]

softmax :: [Float] -> [Float]
softmax xs = 
    let expXs = map exp xs
        sumExp = sum expXs
    in map (/sumExp) expXs

forwardPass :: Network -> Array S Ix2 Float -> ([Array S Ix2 Float], [Array S Ix2 Float], [Array S Ix2 Float], [Array S Ix2 Float], [Float], [Float])
forwardPass Network{..} input = 
    let c1 = convLayer conv1Weights conv1Biases input
        p1 = map maxPool c1
        inputToConv2 = M.makeArray M.Seq (Sz (M.Ix3 (length p1) h w)) (\(M.Ix3 c y x) -> p1 !! c ! Ix2 y x)
            where Sz (Ix2 h w) = M.size (head p1)
        c2 = convLayer3D conv2Weights conv2Biases inputToConv2
        p2 = map maxPool c2
        flattenedP2 = concatMap M.toList p2
        fc = fullyConnected fcWeights fcBiases flattenedP2
        output = softmax $ fullyConnected outputWeights outputBiases fc
    in (c1, p1, c2, p2, fc, output)

drawAtPoint :: (Float, Float) -> Array S Ix2 Float -> Array S Ix2 Float
drawAtPoint (x, y) arr =
    let updatePixel (Ix2 i j) =
            let oldVal = arr ! Ix2 i j
                (cx, cy) = screenToGrid (x, y)
                dist = sqrt ((fromIntegral i - cx)^2 + (fromIntegral j - cy)^2)
                brushValue = max 0 (1 - dist/3)
            in max oldVal brushValue
    in M.makeArray M.Seq (M.size arr) updatePixel :: Array S Ix2 Float
  where
    screenToGrid :: (Float, Float) -> (Float, Float)
    screenToGrid (sx, sy) =
        ( (sx + 400) * 28 / 800
        , (sy + 300) * 28 / 600
        )

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
    in G.translate (-totalWidth/2.0) (-250.0) $ G.pictures
        [G.translate (fromIntegral i * (barWidth + spacing)) 0 $
         G.pictures
            [G.color G.blue $
             G.rectangleSolid barWidth (p * maxHeight),
             G.translate 0 (-30.0) $
             G.scale 0.1 0.1 $
             G.text (show i)]
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
        let (c1, p1, c2, p2, fc, out) = forwardPass network inputImage
            _ = trace ("Conv1 size: " ++ show (M.size (head c1))) $
                trace ("Pool1 size: " ++ show (M.size (head p1))) $
                trace ("Conv2 size: " ++ show (M.size (head c2))) $
                trace ("Pool2 size: " ++ show (M.size (head p2))) $ ()
        in world { conv1Features = c1
                , pool1Features = p1
                , conv2Features = c2
                , pool2Features = p2
                , fcFeatures = fc
                , outputFeatures = out
                , shouldProcess = False
                }
    | isDrawing = let newImage = drawAtPoint mousePos inputImage
                  in world { inputImage = newImage }
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
        Right (_, _, network) -> Right network
  where
    getNetwork = do
        conv1Weights <- replicateM 6 $ getConvKernel 5 5
        conv1Biases <- replicateM 6 getFloatle
        conv2Weights <- replicateM 16 $ getConvKernel3D 6 5 5  -- Adjusted for 3D kernels
        conv2Biases <- replicateM 16 getFloatle
        fcWeights <- getArray2D 120 84
        fcBiases <- replicateM 84 getFloatle
        outputWeights <- getArray2D 84 10
        outputBiases <- replicateM 10 getFloatle
        return Network{..}

main :: IO ()
main = do
    netResult <- loadNetwork "mnist_weights.bin"
    case netResult of
        Left err -> putStrLn $ "Failed to load network: " ++ err
        Right net -> do
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
                    stages = stages,  -- Included stages here
                    stageDescriptions = ["Drawing", "Conv1", "Pool1", "Conv2", "Pool2", "FullyConnected", "Prediction"]
                }
            G.play
                (G.InWindow "CNN Visualization" (800, 600) (100, 100))
                G.white
                30
                world
                render
                handleEvent
                update