package it.unibo.alchemist.model.layers

import it.unibo.alchemist.model.{Environment, Layer, Position}
import me.shadaj.scalapy.py

class ModelLayer[T, P <: Position[P]](
    environment: Environment[T, P],
    private val model: py.Dynamic
) extends Layer[T, P] {
  override def getValue(p: P): T =
    model.asInstanceOf[T]
}
