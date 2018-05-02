package by.andrei6184.fmmodel

/**
  * Implementation of the stochastic gradient descent
  * @param LR learning rate
  * @param LD learning decrease
  * @author Andrei Lysiuk
  */
class SGD(var LR: Double, val LD: Double) {

  def decreaseLR(): Unit = {
    LR *= (1.0 - LD)
  }

  def train(model: Model, samples: Seq[Sample], loss: Model#LossFunction): Double = {
    var quality = 0.0

    samples.foreach { sample =>
      quality += loss.getLoss(sample)
      val coef = LR * loss.getCoef(sample)

      val dBias = loss.dBias(sample)
      val dWeights = loss.dWeights(sample)
      val dPWeights = loss.dPWeights(sample)

      model.updateModel(dBias, dWeights, dPWeights, coef)
    }

    if (loss.C1 != 0.0) {
      val dRegWeights = loss.dRegWeights()
      model.updateWeights(dRegWeights.indices.zip(dRegWeights), LR)
    }
    if (loss.C2 != 0.0) {
      val dRegPWeights = loss.dRegPWeights()
      model.updatePWeights(dRegPWeights.indices.zip(dRegPWeights), LR)
    }

    quality
  }
}
