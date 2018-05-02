package by.andrei6184.fmmodel

import org.apache.commons.cli._

/**
  * Object which implements CLI
  * @author Andrei Lysiuk
  */
object CLI {

  val options: Options = new Options()
    .addOption(Option.builder("f").longOpt("datafile").required().hasArg.argName("file")
      .desc("get data from this file").build())
    .addOption("t", "train", false, "train model")
    .addOption("a", "accumulate the number of the features")
    .addOption("M", true, "feature size = 2^M (default value = 10")
    .addOption("h", "help", false, "print help")
    .addOption("bs", "batchSize", true, "batch size for training (default value = 1)")
    .addOption("i", "numIter", true, "number of iterations used for training (default value = 1)")
    .addOption("d", "depth", true, "model depth (default value = 2)")
    .addOption("LR", "learningRate", true, "initial learning rate (default value = 0.01)")
    .addOption("LD", "learningDecrease", true, "learning decrease factor (default value = 0.01)")
    .addOption("C1", true, "the regularization coefficient of the weights vector (default value = 0.0)")
    .addOption("C2", true, "the regularization coefficient of the weights matrix (default value = 0.0)")
    .addOption("l", "load", true, "load model from file")
    .addOption("s", "save", true, "save model to file")
    .addOption("o", "output", true, "output file with prediction results")
    .addOption("lg", false, "to use logistic regression")
    .addOption("tm", false, "use the timing")

  val parser: DefaultParser = new DefaultParser

  val formatter: HelpFormatter = new HelpFormatter

  def parse(args: Array[String]): scala.Option[Parameters] = {
    if (Set("-h", "--help").subsetOf(args.toSet)) {
      printHelp(); None
    } else {
      try {
        val cl = parser.parse(options, args)
        Some(Parameters(
          path = cl.getOptionValue("f"),
          fSize = 1 << cl.getOptionValue("M", "10").toInt,
          isTrain = cl.hasOption("t"),
          isAccumulative = cl.hasOption("a"),
          batchSize = cl.getOptionValue("bs", "1").toInt,
          numIterations = cl.getOptionValue("i", "1").toInt,
          depth = cl.getOptionValue("d", "2").toInt,
          LR = cl.getOptionValue("LR", "0.01").toDouble,
          LD = cl.getOptionValue("LD", "0.01").toDouble,
          C1 = cl.getOptionValue("C1", "0.0").toDouble,
          C2 = cl.getOptionValue("C2", "0.0").toDouble,
          isLog = cl.hasOption("lg"),
          lFile = cl.getOptionValue("l"),
          sFile = cl.getOptionValue("s"),
          oFile = cl.getOptionValue("o"),
          useTime = cl.hasOption("tm")
        ))
      } catch {
        case _: ParseException => CLI.printHelp(); None
      }
    }
  }

  def printHelp(): Unit = {
    formatter.printHelp("FMModel", options, true)
  }

}
