package it.unibo.experiment

import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.{DataType, Shape}
import ai.djl.nn.core.Linear
import ai.djl.nn.{Activation, Block, SequentialBlock}
object SimpleSequentialDQN {
  def apply(input: Int, hidden: Int, output: Int, manager: NDManager): Block = {
    val blocks = new SequentialBlock()
    blocks.addAll(
      Linear.builder().setUnits(hidden).build(),
      Activation.reluBlock(),
      Linear.builder().setUnits(output).build()
    )
    blocks.initialize(manager, DataType.FLOAT64, new Shape(input))
    blocks
  }
}
