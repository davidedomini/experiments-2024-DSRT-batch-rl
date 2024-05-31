package it.unibo.alchemist.model.learning

trait StateEncoder {
  def encode(state: State): Seq[Double]
}
