package by.andrei6184.fmmodel

/**
  * Class implements the sample of the data
  * @param features contains pairs of the index and count for the feature
  * @param target the target value
  * @author Andrei Lysiuk
  */
case class Sample(features: Map[Int, Int], target: Double)
