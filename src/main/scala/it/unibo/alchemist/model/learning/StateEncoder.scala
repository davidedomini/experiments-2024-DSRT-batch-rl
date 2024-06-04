package it.unibo.alchemist.model.learning

trait StateEncoder[S <: State] {
  def encode(state: S): Seq[Double]
}
