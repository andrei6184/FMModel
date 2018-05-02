package by.andrei6184.fmmodel

import scala.io.Source

/**
  * Class to read samples from input file
  * This class uses hashing trick
  * @author Andrei Lysiuk
  */
object FileReader {
  val DELIMITER = ","
}

/**
  *
  * @param path path to the input file
  * @param fSize max number of features
  * @param isTrain the file will be used for training
  * @param isAccumulative accumulate the count of the features
  */
class FileReader(val path: String, val fSize: Int, val isTrain: Boolean = true,
                 val isAccumulative: Boolean = false) extends Iterator[Sample]{
  private val lines = Source.fromFile(path).getLines()

  override def hasNext: Boolean = lines.hasNext

  override def next(): Sample = {
    val line = lines.next()
    val tokens = line.split(FileReader.DELIMITER).map(_.trim)
    if (isTrain) {
      Sample(hashingTrick(tokens.dropRight(1)), tokens.last.toDouble)
    } else {
      Sample(hashingTrick(tokens), Double.NaN)
    }
  }

  private def hashingTrick(tokens: Seq[String]): Map[Int, Int] = {
    val hashes = tokens.map(t => (t.hashCode & Int.MaxValue) % fSize)
    if (isAccumulative) {
      hashes.groupBy(h => h).map(g => (g._1, g._2.length))
    } else {
      hashes.distinct.map(h => (h, 1)).toMap
    }
  }
}
