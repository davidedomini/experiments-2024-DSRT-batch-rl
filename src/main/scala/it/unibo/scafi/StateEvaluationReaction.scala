package it.unibo.scafi

import it.unibo.alchemist.model.{Environment, GlobalReaction, Node, Position, TimeDistribution}
import it.unibo.alchemist.model.implementations.reactions.AbstractGlobalReaction
import it.unibo.alchemist.model.learning.Molecules
import it.unibo.alchemist.model.molecules.SimpleMolecule

import scala.jdk.CollectionConverters.IteratorHasAsScala

class StateEvaluationReaction[T, P <: Position[P]](
    environment: Environment[T, P],
    distribution: TimeDistribution[T]
  ) extends AbstractGlobalReaction(environment, distribution) {

  override protected def executeBeforeUpdateDistribution(): Unit = {
     nodes.foreach { node =>
       val positions = environment
        .getNeighborhood(node)
        .getNeighbors.iterator().asScala.toList
        .sortBy(neigh => environment.getDistanceBetweenNodes(node, neigh))
        .take(ExperimentParams.neighbors)
        .map { neigh => toPosition2D(neigh) }
       val selfPosition = toPosition2D(node)
       val state = FlockState(selfPosition, positions)
       node.setConcentration(new SimpleMolecule(Molecules.actualState), state.asInstanceOf[T])
    }
  }
  
  private def toPosition2D(node: Node[T]): (Double, Double) = {
    val position = environment.getPosition(node)
    (position.getCoordinate(0), position.getCoordinate(1))
  }

}
