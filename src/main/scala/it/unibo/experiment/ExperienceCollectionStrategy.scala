package it.unibo.experiment

import it.unibo.alchemist.model.{Environment, Position}
import it.unibo.alchemist.model.learning.{Experience, ExperienceBuffer, GlobalExecution, Molecules}
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.experiment.FlockState.stateEncoder
import it.unibo.experiment.ActionSpace.actionEncoder
import org.apache.commons.math3.random.RandomGenerator

import scala.jdk.CollectionConverters.IteratorHasAsScala

class ExperienceCollectionStrategy[T, P <: Position[P]] extends GlobalExecution[T, P] {

  override def execute(environment: Environment[T, P], randomGenerator: RandomGenerator): Unit = {
    nodes(environment).foreach { node =>
      val actualState = node.getConcentration(new SimpleMolecule(Molecules.actualState)).asInstanceOf[FlockState]
      val nextState = node.getConcentration(new SimpleMolecule(Molecules.nextState)).asInstanceOf[FlockState]
      val reward = node.getConcentration(new SimpleMolecule(Molecules.reward)).asInstanceOf[Double]
      val actionID = node.getConcentration(new SimpleMolecule(Molecules.action)).asInstanceOf[Int]
      val action = ActionSpace.all(actionID)
      var experience =
        node.getConcentration(new SimpleMolecule(Molecules.experience)).asInstanceOf[ExperienceBuffer[FlockState]]
      if (experience == null) {
        experience = ExperienceBuffer(ExperimentParams.experienceBufferMaxSize)
      }
      experience.insert(Experience(actualState, action, reward, nextState))
      node.setConcentration(new SimpleMolecule(Molecules.experience), experience.asInstanceOf[T])
    }
  }

  private def nodes(environment: Environment[T, P]) =
    environment.getNodes.iterator().asScala.toList
}
