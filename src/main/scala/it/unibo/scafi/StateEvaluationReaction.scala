package it.unibo.scafi

import it.unibo.alchemist.model.{Environment, GlobalReaction, Node, Position, TimeDistribution}
import it.unibo.alchemist.model.implementations.reactions.AbstractGlobalReaction
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
        .take(5) // TODO - magic number
        .map { neigh => environment.getPosition(neigh) }

       // TODO - create a state from positions
       // TODO - take molecule name from molecules
       node.setConcentration(new SimpleMolecule("NeighPositions"), positions.asInstanceOf[T])

    }
  }

}
