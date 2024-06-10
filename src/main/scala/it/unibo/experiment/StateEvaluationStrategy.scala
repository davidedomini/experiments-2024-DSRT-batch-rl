package it.unibo.experiment

import it.unibo.alchemist.model.{Environment, Node}
import it.unibo.alchemist.model.learning.{GlobalExecution, Molecules}
import it.unibo.alchemist.model.molecules.SimpleMolecule

import scala.jdk.CollectionConverters.IteratorHasAsScala

class StateEvaluationStrategy(actualState: Boolean) extends GlobalExecution {

  override def execute(environment: Environment[Any, Nothing]): Unit = {
    nodes(environment).foreach { node =>
      val positions = environment
        .getNeighborhood(node)
        .getNeighbors.iterator().asScala.toList
        .sortBy(neigh => environment.getDistanceBetweenNodes(node, neigh))
        .take(ExperimentParams.neighbors)
        .map { neigh => toPosition2D(neigh, environment) }
      val selfPosition = toPosition2D(node, environment)
      val state = FlockState(selfPosition, positions)
      val encodedState = FlockState.stateEncoder.encode(state)
      storeState(node, state, encodedState)
    }
  }

  private def toPosition2D(node: Node[Any], environment: Environment[Any, Nothing]): (Double, Double) = {
    val position = environment.getPosition(node)
    //(position.getCoordinate(0), position.getCoordinate(1))
    // TODO - fix
    (0.0, 0.0)
  }

  private def storeState(node: Node[Any], state: FlockState, encodedState: Seq[Double]): Unit = {
    if(actualState){
      node.setConcentration(new SimpleMolecule(Molecules.actualState), state)
      node.setConcentration(new SimpleMolecule(Molecules.encodedActualState), encodedState)
    } else {
      node.setConcentration(new SimpleMolecule(Molecules.nextState), state)
      node.setConcentration(new SimpleMolecule(Molecules.encodedNextState), encodedState)
    }
  }

  private def nodes(environment: Environment[Any, Nothing]) =
    environment.getNodes.iterator().asScala.toList
}
