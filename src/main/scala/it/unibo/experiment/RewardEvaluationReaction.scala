package it.unibo.experiment

import it.unibo.alchemist.model.{Environment, Position, TimeDistribution}
import it.unibo.alchemist.model.implementations.reactions.AbstractGlobalReaction
import it.unibo.alchemist.model.learning.{Action, Molecules}
import it.unibo.alchemist.model.molecules.SimpleMolecule

class RewardEvaluationReaction [T, P <: Position[P]] (
    environment: Environment[T, P],
    distribution: TimeDistribution[T]
  ) extends AbstractGlobalReaction(environment, distribution) {

  override protected def executeBeforeUpdateDistribution(): Unit = {
    nodes.foreach { node =>
      val state = node.getConcentration(new SimpleMolecule(Molecules.nextState)).asInstanceOf[FlockState]
      val reward = computeReward(state)
      node.setConcentration(new SimpleMolecule(Molecules.reward), reward.asInstanceOf[T])
    }
  }

  private def computeReward(state: FlockState): Double = {
    val distances = toDistances(state)
    cohesion(distances) + collision(distances)
  }

  private def cohesion(distances: Seq[Double]): Double = {
    val maxDistance = distances.max
    if (maxDistance > ExperimentParams.targetDistance) {
      -(maxDistance - ExperimentParams.targetDistance)
    } else {
      0.0
    }
  }

  private def collision(distances: Seq[Double]): Double = {
    val minDistance = distances.min
    if (minDistance < ExperimentParams.targetDistance) {
      2 * math.log(minDistance / ExperimentParams.targetDistance)
    }
    else {
      0.0
    }
  }

  private def toDistances(state: FlockState): Seq[Double] = {
    val (selfX, selfY) =  state.myPosition
    state.neighborsPosition.map { case (x,y) =>
      Math.sqrt(Math.pow(selfX - x, 2) + Math.pow(selfY - y, 2))
    }
  }

}
