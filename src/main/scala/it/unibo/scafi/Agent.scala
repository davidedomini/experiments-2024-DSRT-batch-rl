package it.unibo.scafi

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._

class Agent
  extends AggregateProgram
  with StandardSensors
  with ScafiAlchemistSupport {

  override def main(): Unit = {
    node.put("Counter", mid())
  }

}
