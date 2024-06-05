package it.unibo.experiment

import it.unibo.alchemist.model.implementations.reactions.AbstractGlobalReaction
import it.unibo.alchemist.model.learning.{Experience, ExperienceBuffer, Molecules, State}
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{Environment, Position, TimeDistribution}
import it.unibo.experiment.FlockState.stateEncoder
import it.unibo.experiment.ActionSpace.actionEncoder

class ExperienceCollectionReaction [T, P <: Position[P]] (
    environment: Environment[T, P],
    distribution: TimeDistribution[T]
  ) extends AbstractGlobalReaction(environment, distribution) {

  override protected def executeBeforeUpdateDistribution(): Unit = {
    nodes.foreach { node =>
      val actualState = node.getConcentration(new SimpleMolecule(Molecules.actualState)).asInstanceOf[FlockState]
      val nextState = node.getConcentration(new SimpleMolecule(Molecules.nextState)).asInstanceOf[FlockState]
      val reward = node.getConcentration(new SimpleMolecule(Molecules.reward)).asInstanceOf[Double]
      val actionID = node.getConcentration(new SimpleMolecule(Molecules.action)).asInstanceOf[Int]
      val action = ActionSpace.all(actionID)
      val experience = node.getConcentration(new SimpleMolecule(Molecules.experience)).asInstanceOf[ExperienceBuffer[FlockState]]
      experience.insert(Experience(actualState, action, reward, nextState))
      node.setConcentration(new SimpleMolecule(Molecules.experience), experience.asInstanceOf[T])
    }
  }
 
}
