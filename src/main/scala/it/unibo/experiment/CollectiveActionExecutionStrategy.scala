package it.unibo.experiment

import it.unibo.alchemist.model.{Environment, Node, Position}
import it.unibo.alchemist.model.learning.{Action, GlobalExecution, Molecules}
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.experiment.ActionSpace.{East, North, NorthEast, NorthWest, South, SouthEast, SouthWest, West}

import scala.jdk.CollectionConverters.IteratorHasAsScala

class CollectiveActionExecutionStrategy[T, P <: Position[P]] extends GlobalExecution[T, P] {

  override def execute(environment: Environment[T, P]): Unit = {
    nodes(environment).foreach { node =>
      val actionID = node.getConcentration(new SimpleMolecule(Molecules.action)).asInstanceOf[Int]
      val action = ActionSpace.all(actionID)
      val newPosition = computeNewPosition(node, action, environment)
      environment.moveNodeToPosition(node, newPosition)
    }
  }

  private def computeNewPosition(node: Node[T], action: Action, environment: Environment[T, P]): P = {
    if (action != null) {
      action match {
        case North =>
          environment.getPosition(node).plus(Array(0.0, ExperimentParams.deltaMovement))
        case South =>
          environment.getPosition(node).plus(Array(0.0, -ExperimentParams.deltaMovement))
        case East =>
          environment.getPosition(node).plus(Array(ExperimentParams.deltaMovement, 0.0))
        case West =>
          environment.getPosition(node).plus(Array(-ExperimentParams.deltaMovement, 0.0))
        case NorthEast =>
          environment.getPosition(node).plus(Array(ExperimentParams.deltaMovement, ExperimentParams.deltaMovement))
        case SouthEast =>
          environment.getPosition(node).plus(Array(ExperimentParams.deltaMovement, -ExperimentParams.deltaMovement))
        case NorthWest =>
          environment.getPosition(node).plus(Array(-ExperimentParams.deltaMovement, ExperimentParams.deltaMovement))
        case SouthWest =>
          environment.getPosition(node).plus(Array(-ExperimentParams.deltaMovement, -ExperimentParams.deltaMovement))
      }
    } else {
      environment.getPosition(node)
    }
  }

  private def nodes(environment: Environment[T, P]) =
    environment.getNodes.iterator().asScala.toList
}
