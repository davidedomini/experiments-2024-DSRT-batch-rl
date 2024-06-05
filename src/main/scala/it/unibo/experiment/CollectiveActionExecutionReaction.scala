package it.unibo.experiment
import it.unibo.alchemist.model.implementations.reactions.AbstractGlobalReaction
import it.unibo.alchemist.model.learning.{Action, Molecules}
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{Environment, Node, Position, TimeDistribution}
import it.unibo.experiment.ActionSpace._

class CollectiveActionExecutionReaction[T, P <: Position[P]] (
    environment: Environment[T, P],
    distribution: TimeDistribution[T]
  ) extends AbstractGlobalReaction(environment, distribution) {

  override protected def executeBeforeUpdateDistribution(): Unit = {
    nodes.foreach { node =>
      val actionID = node.getConcentration(new SimpleMolecule(Molecules.action)).asInstanceOf[Int]
      val action = ActionSpace.all(actionID)
      val newPosition = computeNewPosition(node, action)
      environment.moveNodeToPosition(node, newPosition)
    }
  }

  private def computeNewPosition(node: Node[T], action: Action): P = {
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

}
