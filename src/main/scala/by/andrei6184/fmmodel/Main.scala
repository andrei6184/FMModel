package by.andrei6184.fmmodel

import java.io.PrintWriter

/**
  * Object which implements the main work pipeline
  * @author Andrei Lysiuk
  */
object Main {

  def train(fileReader: FileReader, batchSize: Int, numIterations: Int, sgd: SGD, model: Model, loss: Model#LossFunction): Unit = {
    var count = 0
    var totalQuality = 0.0
    var totalCount = 0

    while(fileReader.hasNext) {
      val samples = fileReader.take(batchSize).toArray
      var iterQuality = 0.0
      1.to(numIterations).foreach{_ =>
        iterQuality += sgd.train(model, scala.util.Random.shuffle(samples), loss)
      }
      sgd.decreaseLR()
      val iterCount = numIterations * samples.length
      totalQuality += iterQuality
      totalCount += iterCount
      count += samples.length

      println(s"The quality after processing $count samples: ${iterQuality / iterCount} ${totalQuality / totalCount}")
    }

  }

  def test(fileReader: FileReader, model: Model): Seq[Double] = {
    val results = fileReader.map(model.predict).toSeq
    results
  }

  def saveResults(outputPath: String, results: Seq[Double]): Unit = {
    val out = new PrintWriter(outputPath)
    results.foreach(out.println)
    out.close()
  }

  def main(args: Array[String]): Unit = {
    val parameters = CLI.parse(args)

    parameters.foreach { p =>
      val startTime = System.nanoTime()
      val fileReader = new FileReader(p.path, p.fSize, p.isTrain, p.isAccumulative)
      if (p.isTrain) {
        println("Start training")
        val sgd = new SGD(p.LR, p.LD)
        val model = if (p.lFile != null) Model.load(p.lFile) else new Model(p.fSize, p.depth)
        val loss = if (p.isLog) new model.LogLossFunction(p.C1, p.C2) else new model.MSELossFunction(p.C1, p.C2)
        train(fileReader, p.batchSize, p.numIterations, sgd, model, loss)
        if (p.sFile != null) Model.save(p.sFile, model)
        println("Stop training")
      } else {
        if (p.lFile == null || p.oFile == null) {
          scala.Console.err.println("You must use -l option to load model and -o to save the results")
          sys.exit(1)
        }
        println("Start testing")
        val model = Model.load(p.lFile)
        val results = test(fileReader, model)
        saveResults(p.oFile, results)
        println("Stop testing")
      }
      val endTime = System.nanoTime()
      if (p.useTime) println(s"Total time is ${(endTime - startTime) / 1e6} ms")
    }
  }
}
