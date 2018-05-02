package by.andrei6184.fmmodel

import java.io._


/**
  * Factorization machine implementation
  * @author Andrei Lysiuk
  */
object Model {

  def save(filePath: String, model: Model): Unit = {
    val outputStream = new ObjectOutputStream(new FileOutputStream(filePath))
    outputStream.writeObject(model)
  }

  def load(filePath: String): Model = {
    val inputStream = new ObjectInputStream(new FileInputStream(filePath))
    inputStream.readObject().asInstanceOf[Model]
  }

}

/**
  * @param fSize the count of the features
  * @param depth the depth of the model
  */
class Model(val fSize: Int, val depth: Int) extends Serializable {
  private var bias: Double = 0.0
  private val weights: Array[Double] = Array.ofDim[Double](fSize)
  private val pWeights: Array[Array[Double]] = Array.ofDim[Double](fSize, depth)

  abstract class LossFunction(val C1: Double, val C2: Double) {

    def getCoef(sample: Sample): Double

    def getLoss(sample: Sample): Double

    def dBias(sample: Sample): Double = 1.0

    def dWeights(sample: Sample): Seq[(Int, Double)] = {
      sample.features.mapValues(_.toDouble).toSeq
    }

    def dPWeights(sample: Sample): Seq[(Int, Array[Double])] = {
      val tmpSum = Array.ofDim[Double](depth)
      sample.features.foreach { case (i, x) =>
        0.until(depth).foreach(j => tmpSum(j) += x * pWeights(i)(j))
      }
      sample.features.map{ case (i, x) =>
        (i, 0.until(depth).map(j => x * (tmpSum(j) - x * pWeights(i)(j))).toArray)
      }.toSeq
    }

    def dRegWeights(): Array[Double] = {
      weights.map(_ * C1)
    }

    def dRegPWeights(): Array[Array[Double]] = {
      pWeights.map(_.map(_ * C2))
    }

    def regLoss: Double = {
      var loss = 0.0
      if (C1 != 0) {
        loss += C1 / 2.0 * dot(weights, weights)
      }
      if (C2 != 0) {
        loss += C2 / 2.0 * pWeights.map(w => dot(w, w)).sum
      }
      loss
    }

  }

  class MSELossFunction(C1: Double, C2: Double) extends LossFunction(C1, C2) {
    override def getCoef(sample: Sample): Double = {
      2.0 * (predict(sample) - sample.target)
    }

    override def getLoss(sample: Sample): Double = {
      math.pow(predict(sample) - sample.target, 2.0) + regLoss
    }
  }

  class LogLossFunction(C1: Double, C2: Double) extends LossFunction(C1, C2) {

    def getMargin(sample: Sample): Double ={
      sample.target * predict(sample)
    }

    override def getCoef(sample: Sample): Double = {
      -sample.target / (1.0 + math.exp(getMargin(sample)))
    }

    override def getLoss(sample: Sample): Double = {
      math.log1p(math.exp(-getMargin(sample))) + regLoss
    }
  }

  def predict(sample: Sample): Double = {
    val tmpSum = Array.ofDim[Double](depth)
    val linearPart = sample.features.map { case (i, x) =>
      0.until(depth).foreach(j => tmpSum(j) += x * pWeights(i)(j))
      x * weights(i) - 0.5 * dot(pWeights(i), pWeights(i))
    }.sum
    bias + linearPart + 0.5 * dot(tmpSum, tmpSum)
  }

  def updateBias(dBias: Double, coef: Double): Unit = {
    bias -= coef * dBias
  }

  def updateWeights(dWeights: Seq[(Int, Double)], coef: Double): Unit = {
    dWeights.foreach{ case (i, dW) => weights(i) -= coef * dW }
  }

  def updatePWeights(dPWeights: Seq[(Int, Array[Double])], coef: Double): Unit = {
    dPWeights.foreach{ case (i, dPW) =>
      0.until(depth).foreach(j => pWeights(i)(j) -= coef * dPW(j))
    }
  }

  def updateModel(dBias: Double, dWeights: Seq[(Int, Double)], dPWeights: Seq[(Int, Array[Double])],
                  coef: Double): Unit = {
    updateBias(dBias, coef)
    updateWeights(dWeights, coef)
    updatePWeights(dPWeights, coef)
  }

  private   def dot(fVector: Seq[Double], sVector: Seq[Double]): Double = {
    fVector.iterator.zip(sVector.iterator).map{case (x, y) => x * y}.sum
  }

}
