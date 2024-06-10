package it.unibo.experiment

import it.unibo.alchemist.model.Environment
import it.unibo.alchemist.model.learning.{Experience, ExperienceBuffer, GlobalExecution, Molecules}
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.experiment.FlockState.stateEncoder
import it.unibo.experiment.ActionSpace.actionEncoder
import scala.jdk.CollectionConverters.IteratorHasAsScala

class ExperienceCollectionStrategy extends GlobalExecution {

  override def execute(environment: Environment[Any, Nothing]): Unit = {
    nodes(environment).foreach { node =>
      val actualState = node.getConcentration(new SimpleMolecule(Molecules.actualState)).asInstanceOf[FlockState]
      val nextState = node.getConcentration(new SimpleMolecule(Molecules.nextState)).asInstanceOf[FlockState]
      val reward = node.getConcentration(new SimpleMolecule(Molecules.reward)).asInstanceOf[Double]
      val actionID = node.getConcentration(new SimpleMolecule(Molecules.action)).asInstanceOf[Int]
      val action = ActionSpace.all(actionID)
      var experience = node.getConcentration(new SimpleMolecule(Molecules.experience)).asInstanceOf[ExperienceBuffer[FlockState]]
      if(experience == null){
        experience = ExperienceBuffer(ExperimentParams.experienceBufferMaxSize)
      }
      experience.insert(Experience(actualState, action, reward, nextState))
      node.setConcentration(new SimpleMolecule(Molecules.experience), experience)
    }
  }

  private def nodes(environment: Environment[Any, Nothing]) =
    environment.getNodes.iterator().asScala.toList
}
