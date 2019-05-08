
import CodingFunctions
import UtilsPlot

N = 1000
K = 3
# (ModFs,DemodFs) = CodingFunctions.GetCosCos(N = N, K = K)
#(ModFs,DemodFs) = CodingFunctions.GetBinaryFunc(N = N, K = K)
(ModFs,DemodFs) = CodingFunctions.GetSquare(N = N, K = K)
#(ModFs,DemodFs) = CodingFunctions.GetHamK3(N = N)
# (ModFs,DemodFs) = CodingFunctions.GetHamK4(N = N)
# (ModFs,DemodFs) = CodingFunctions.GetHamK5(N = N)
#(ModFs,DemodFs) = CodingFunctions.GetMultiFreqCosK5(N = N)


UtilsPlot.PlotCodingScheme(ModFs,DemodFs)



