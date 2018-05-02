package by.andrei6184.fmmodel

/**
  * Input parameters of the program
  * @author Andrei Lysiuk
  */
case class Parameters(path: String, fSize: Int, isTrain: Boolean, isAccumulative: Boolean,
                      batchSize: Int, numIterations: Int, depth: Int, LR: Double, LD: Double,
                      C1: Double, C2: Double, isLog: Boolean, lFile: String, sFile: String, oFile: String, useTime: Boolean)
