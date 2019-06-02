import numpy as np
import scipy.io as sio
import CodingFunctions
import UtilsPlot
import Utils

N = 10000
K = 3
#(ModFs,DemodFs) = CodingFunctions.GetCodingFromFile('coding_functions.npz')
#(ModFs,DemodFs) = CodingFunctions.GetCosCos(N = N, K = K)
#(ModFs,DemodFs) = CodingFunctions.GetBinaryFunc(N = N, K = K)
#(ModFs,DemodFs) = CodingFunctions.GetSquare(N = N, K = K)
(ModFs,DemodFs) = CodingFunctions.GetHamK3(N = N)
# (ModFs,DemodFs) = CodingFunctions.GetHamK4(N = N)
# (ModFs,DemodFs) = CodingFunctions.GetHamK5(N = N)
#(ModFs,DemodFs) = CodingFunctions.GetMultiFreqCosK5(N = N)


UtilsPlot.PlotCodingScheme(ModFs,DemodFs)
CorrFs = Utils.GetCorrelationFunctions(ModFs, DemodFs)
sio.savemat('Ham_coding_functions.mat', {'ModFs':ModFs, 'DemodFs':DemodFs, 'CorrFs':CorrFs})
