{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}

module Main where

import qualified Graphics.Gloss as G
import qualified Graphics.Gloss.Interface.Pure.Game as G
import qualified Data.Massiv.Array as M
import Data.Massiv.Array (Array, Comp(..), B, D, S, Ix2(..), (!), Sz(..))
import qualified Data.Massiv.Array.Numeric as M
import Control.Monad (forM_, replicateM)
import System.Random (randomRIO)
import qualified Data.Vector.Unboxed as V
import qualified Data.ByteString.Lazy as BS
import Data.Binary
import Data.Binary.Get
import System.IO.Error (catchIOError)

-- Neural Network Types
type Weight = Float
type Bias = Float
type Layer = Array S Ix2 Float

data Network = Network 
    { conv1Weights :: [Array S Ix2 Weight]
    , conv1Biases :: [Bias]
    , conv2Weights :: [Array S Ix2 Weight]
    , conv2Biases :: [Bias]
    , fcWeights :: Array S Ix2 Weight
    , fcBiases :: [Bias]
    , outputWeights :: Array S Ix2 Weight
    , outputBiases :: [Bias]
    }

data World = World
    { inputImage :: Array S Ix2 Float
    , conv1Features :: [Array S Ix2 Float]
    , conv2Features :: [Array S Ix2 Float]
    , fcFeatures :: [Float]
    , outputFeatures :: [Float]
    , network :: Network
    , currentStage :: Stage
    , mousePos :: (Float, Float)
    , isDrawing :: Bool
    }

data Stage = Drawing | Conv1 | Conv2 | FullyConnected | Prediction
    deriving (Eq, Show)

-- Neural Network Operations
relu :: Float -> Float
relu x = max 0 x

convolve :: Array S Ix2 Float -> Array S Ix2 Float -> Array S Ix2 Float
convolve input kernel =
    let M.Sz (Ix2 h w) = M.size input
        M.Sz (Ix2 kh kw) = M.size kernel
        halfH = kh `div` 2
        halfW = kw `div` 2
        get_pixel i j
            | i >= 0 && i < h && j >= 0 && j < w = input ! Ix2 i j
            | otherwise = 0
        get_kernel i j = kernel ! Ix2 i j
    in M.makeArray M.Seq (M.Sz (Ix2 h w)) $ \(Ix2 i j) ->
        sum [ get_pixel (i + ki - halfH) (j + kj - halfW) * get_kernel ki kj
            | ki <- [0..kh-1]
            , kj <- [0..kw-1]
            , let i' = i + ki - halfH
            , let j' = j + kj - halfW
            , i' >= 0, i' < h
            , j' >= 0, j' < w
            ]

convLayer :: [Array S Ix2 Float] -> [Bias] -> Array S Ix2 Float -> [Array S Ix2 Float]
convLayer weights biases input = 
    zipWith (\w b -> M.computeS $ M.map (relu . (+b)) $ convolve input w) weights biases

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

forwardPass :: Network -> Array S Ix2 Float -> ([Array S Ix2 Float], [Array S Ix2 Float], [Float], [Float])
forwardPass Network{..} input = 
    let c1 = convLayer conv1Weights conv1Biases input
        c2 = concatMap (convLayer conv2Weights conv2Biases) c1
        flattenedC2 = concatMap M.toList c2
        fc = fullyConnected fcWeights fcBiases flattenedC2
        output = softmax $ fullyConnected outputWeights outputBiases fc
    in (c1, c2, fc, output)

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
        , (300 - sy) * 28 / 600
        )

renderDrawingStage :: Array S Ix2 Float -> G.Picture
renderDrawingStage img = 
    G.scale 20 20 $ G.pictures 
        [G.translate (fromIntegral x) (fromIntegral y) $ 
         G.color (G.makeColor 0 0 0 (img ! Ix2 y x)) $
         G.rectangleSolid 1 1
        | x <- [0..27], y <- [0..27]]

renderFeatureMaps :: [Array S Ix2 Float] -> G.Picture
renderFeatureMaps features = 
    let n = length features
        cols = ceiling (sqrt (fromIntegral n :: Double)) :: Int
        rows = ceiling ((fromIntegral n :: Double) / fromIntegral cols) :: Int
        scale = min (600 / fromIntegral rows) 
                   (800 / fromIntegral cols)
        positions = [(fromIntegral x, fromIntegral y) | y <- [0..rows-1], x <- [0..cols-1]]
    in G.pictures 
        [G.translate (x * scale - 400) (y * scale - 300) $
         G.scale (scale/28) (scale/28) $
         renderFeatureMap f
        | (f, (x, y)) <- zip features positions]

renderFeatureMap :: Array S Ix2 Float -> G.Picture
renderFeatureMap feature =
    G.pictures 
        [G.translate (fromIntegral x) (fromIntegral y) $
         G.color (G.makeColor v v v 1) $
         G.rectangleSolid 1 1
        | x <- [0..27], y <- [0..27]
        , let v = min 1 $ max 0 $ feature ! Ix2 y x]

renderFullyConnected :: [Float] -> G.Picture
renderFullyConnected activations =
    let n = length activations
        width = 800
        height = 600
        spacing = width / fromIntegral n
    in G.pictures 
        [G.translate (x * spacing - width/2) 0 $
         G.color (G.makeColor v v v 1) $
         G.circleSolid 10
        | (v, x) <- zip activations [0..]]

renderPrediction :: [Float] -> G.Picture
renderPrediction probs =
    let barWidth = 60
        spacing = 20
        totalWidth = (barWidth + spacing) * 10
        maxHeight = 500
    in G.translate (-totalWidth/2) (-250) $ G.pictures
        [G.translate (fromIntegral i * (barWidth + spacing)) 0 $
         G.pictures
            [G.color G.blue $
             G.rectangleSolid barWidth (p * maxHeight),
             G.translate 0 (-30) $
             G.scale 0.1 0.1 $
             G.text (show i)]
        | (p, i) <- zip probs [0..9]]

handleEvent (G.EventKey (G.Char key) G.Down _ _) world@World{..} =
    case key of
        '1' -> world { currentStage = Drawing }
        '2' -> world { currentStage = Conv1 }
        '3' -> world { currentStage = Conv2 }
        '4' -> world { currentStage = FullyConnected }
        '5' -> world { currentStage = Prediction }
        'c' -> world { inputImage = M.makeArray M.Seq (M.Sz (Ix2 28 28)) (const 0) :: Array S Ix2 Float }
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
update _ world@World{..} =
    if isDrawing
    then let newImage = drawAtPoint mousePos inputImage
             (c1, c2, fc, out) = forwardPass network newImage
         in world { inputImage = newImage
                 , conv1Features = c1
                 , conv2Features = c2
                 , fcFeatures = fc
                 , outputFeatures = out
                 }
    else world

render :: World -> G.Picture
render World{..} = case currentStage of
    Drawing -> renderDrawingStage inputImage
    Conv1 -> renderFeatureMaps conv1Features
    Conv2 -> renderFeatureMaps conv2Features
    FullyConnected -> renderFullyConnected fcFeatures
    Prediction -> renderPrediction outputFeatures

getConvKernel :: Int -> Int -> Get (Array S Ix2 Float)
getConvKernel h w = do
    values <- replicateM (h * w) getFloatle
    return $ M.makeArray M.Seq (M.Sz (Ix2 h w)) $ \(Ix2 i j) ->
        values !! (i * w + j)

getArray2D :: Int -> Int -> Get (Array S Ix2 Float)
getArray2D h w = do
    values <- replicateM (h * w) getFloatle
    return $ M.makeArray M.Seq (M.Sz (Ix2 h w)) $ \(Ix2 i j) ->
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
        conv2Weights <- replicateM 16 $ getConvKernel 5 5
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
            let initialImage = M.makeArray M.Seq (M.Sz (Ix2 28 28)) (const 0) :: Array S Ix2 Float
                world = World {
                    inputImage = initialImage,
                    conv1Features = [],
                    conv2Features = [],
                    fcFeatures = [],
                    outputFeatures = replicate 10 0.1,
                    network = net,
                    currentStage = Drawing,
                    mousePos = (0, 0),
                    isDrawing = False
                }
            G.play
                (G.InWindow "CNN Visualization" (800, 600) (100, 100))
                G.white
                30
                world
                render
                handleEvent
                update