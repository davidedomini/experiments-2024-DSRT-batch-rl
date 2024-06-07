package it.unibo.alchemist.model.layers

import it.unibo.alchemist.model.{Environment, Layer, Position}
import me.shadaj.scalapy.py

class ModelLayer[P <: Position[P]](
    environment: Environment[_, P],
    private val model: py.Dynamic
  ) extends Layer[py.Dynamic, P] {
  override def getValue(p: P): py.Dynamic = {
    model
  }
}