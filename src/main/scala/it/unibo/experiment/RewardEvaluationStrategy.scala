package it.unibo.experiment

import it.unibo.alchemist.model.{Environment, Position}
import it.unibo.alchemist.model.learning.{GlobalExecution, Molecules}
import it.unibo.alchemist.model.molecules.SimpleMolecule

import scala.jdk.CollectionConverters.IteratorHasAsScala

class RewardEvaluationStrategy[T, P <: Position[P]] extends GlobalExecution[T, P]{

  override def execute(environment: Environment[T, P]): Unit = {
    nodes(environment).foreach { node =>
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
    val neighbors = fixIfEmptyNeighborhood(state.neighborsPosition)
    neighbors.map { case (x,y) =>
      Math.sqrt(Math.pow(selfX - x, 2) + Math.pow(selfY - y, 2))
    }
  }

  private def fixIfEmptyNeighborhood(positions: Seq[(Double, Double)]): Seq[(Double, Double)] = {
    val fill = List.fill(ExperimentParams.neighbors)((0.0, 0.0))
    (positions ++ fill).take(ExperimentParams.neighbors)
  }

  private def nodes(environment: Environment[T , P]) =
    environment.getNodes.iterator().asScala.toList

}
