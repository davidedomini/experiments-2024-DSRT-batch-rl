package it.unibo.alchemist.model.learning

trait ActionEncoder{
  def encode(action: Action): Int
}
