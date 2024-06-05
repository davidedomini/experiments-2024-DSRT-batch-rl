package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.learning.Molecules
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{Environment, Position, TimeDistribution}
import it.unibo.interop.PythonModules._
import me.shadaj.scalapy.py.SeqConverters

class ActionChoiceReaction[T, P <: Position[P]] (
    environment: Environment[T, P],
    distribution: TimeDistribution[T]
  ) extends AbstractGlobalReaction(environment, distribution) {

  override protected def executeBeforeUpdateDistribution(): Unit = {
    val observations = nodes
      .map { node => node.getConcentration(new SimpleMolecule(Molecules.encodedActualState)).asInstanceOf[List[Double]] }
    // TODO - implement in python
    val actions = observations.map { o => 1 }//pythonUtils.actions_inference(observations.toPythonProxy).asInstanceOf[List[Int]]
    actions
      .zipWithIndex
      .foreach { case (action, index) =>
        environment.getNodeByID(index).setConcentration(new SimpleMolecule(Molecules.action), action.asInstanceOf[T])
      }
  }

}
