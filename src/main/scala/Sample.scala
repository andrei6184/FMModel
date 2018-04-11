object Util {

  type Vector = Array[Double]

  def dot(first: Vector, second: Vector): Double = {
    first.iterator.zip(second.iterator).map(x => x._1 * x._2).sum
  }

  def mult(vector: Vector, coef: Double): Vector = {
    vector.map(coef*_)
  }

  def plus(first: Vector, second: Vector): Vector = {
    first.iterator.zip(second.iterator).map(x => x._1 + x._2).toArray
  }

  def minus(first: Vector, second: Vector): Vector = {
    first.iterator.zip(second.iterator).map(x => x._1 - x._2).toArray
  }

}

import Util._

case class Sample(indexes: Seq[Int], target: Double)

case class DataSet(samples: Seq[Sample]) {
  def length: Int = samples.length
}

case class FileReader(featureSize: Int) {
  val DELIMITER = ","

  def hashingTrick(CF: String): Int = CF.hashCode % featureSize

  def readDataSet(path: String, isTrain: Boolean = true): DataSet = {
    val source = io.Source.fromFile(path)
    val samples = source.getLines.map { line =>
      val tokens = line.split(DELIMITER)
      if (isTrain) {
        Sample(tokens.dropRight(1).map(hashingTrick), tokens.last.toDouble)
      } else {
        Sample(tokens.map(_.toInt), Double.NaN)
      }
    }.toSeq
    DataSet(samples)
  }
}

object Model {
  import java.io._

  def save(path: String, model: Model): Unit = {
    val outputStream = new ObjectOutputStream(new FileOutputStream(path))
    outputStream.writeObject(model)
  }

  def load(path: String): Model = {
    val inputStream = new ObjectInputStream(new FileInputStream(path))
    inputStream.readObject().asInstanceOf[Model]
  }

}

class Model(val size: Int, val deep: Int) extends Serializable {

  var bias: Double = 0.0
  var weights: Vector = Array.ofDim[Double](size)
  var pairWeights: Array[Vector] = Array.ofDim[Double](size, deep)

  def predict(sample: Sample): Double = {
    var result = bias
    val tempSum = Array.ofDim[Double](deep)
    result += sample.indexes.map { i =>
      for (j <- 0.until(deep)) {
        tempSum(j) += pairWeights(i)(j)
      }
      weights(i) - 0.5 * dot(pairWeights(i), pairWeights(i))
    }.sum

    result + 0.5 * dot(tempSum, tempSum)
  }

  def update(dBias: Double, dWeights: Seq[(Int, Double)], dPairWeights: Seq[(Int, Vector)]): Unit = {
    bias += dBias
    dWeights.foreach{case (i, v) => weights(i) += v}
    dPairWeights.foreach{case (i, v) => 0.until(deep).foreach(j => pairWeights(i)(j) += v(j))}
  }


}

object MSE {
  def error(model: Model, dataSet: DataSet): Double = {
    val errors = dataSet.samples.map(s => math.pow(model.predict(s) - s.target, 2))
    errors.sum / errors.length
  }
}

class SGD(LR: Double, iterNum: Int) {

  def train(model: Model, dataSet: DataSet): Unit = {
    println(s"Start error = ${MSE.error(model, dataSet)}")
    var lr = LR
    for (it <- 1.to(iterNum)) {
      dataSet.samples.foreach { sample =>
        val coef = 2 * (sample.target - model.predict(sample)) * lr
        val dBias = coef
        val dWeigths = sample.indexes.map(i => (i, coef))

        val tempSum = Array.ofDim[Double](model.deep)
        sample.indexes.foreach { i =>
          for (j <- 0.until(model.deep)) {
            tempSum(j) += model.pairWeights(i)(j)
          }
        }

        val dPairWeights = sample.indexes.map { i =>
          (i, minus(tempSum, model.pairWeights(i)).map(coef * _))
        }

        model.update(dBias, dWeigths, dPairWeights)
      }
      println(s"After $it iteration error = ${MSE.error(model, dataSet)}")
      lr /= 1.1
    }
  }

}


object MainProcess {

  def main(args: Array[String]): Unit = {
    val featureSize = 1 << 10
    val deep = 10
    val fileReader = FileReader(featureSize)
    val dataSet = fileReader.readDataSet("data/train-20m.csv")
    val model = new Model(featureSize, deep)
    val sgd = new SGD(0.1, 200)
    sgd.train(model, dataSet)
    Model.save("models/model-20m.mdl", model)
//    val model = Model.load("model-100k.mdl")
//    val dataSet = FileReader(model.size).readDataSet("data/test-100k.csv")
//    println(new SGD(0.1).error(model, dataSet))
  }

}
