package it.unibo.experiment

import it.unibo.alchemist.model.learning.{GlobalExecution, Molecules}
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{Environment, Position}
import it.unibo.interop.PythonModules.torch
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

import scala.jdk.CollectionConverters.IteratorHasAsScala

class ActionChoiceStrategy[T, P <: Position[P]] extends GlobalExecution[T, P]{

  override def execute(environment: Environment[T, P]): Unit = {
    val policy = loadNN(environment)

    val observations = nodes(environment)
      .map { node => node.getConcentration(new SimpleMolecule(Molecules.encodedActualState)).asInstanceOf[List[Double]] }
    val observationsTensor = torch.Tensor(observations.toPythonProxy)
    val qValues = policy(observationsTensor)
    val actions = qValues.argmax(dim=1).tolist().as[List[Int]]
    actions
      .zipWithIndex
      .foreach { case (action, index) =>
        environment.getNodeByID(index).setConcentration(new SimpleMolecule(Molecules.action), action.asInstanceOf[T])
      }
  }

  private def loadNN(environment: Environment[T , P]): py.Dynamic = {
    nodes(environment).head.getConcentration(new SimpleMolecule(Molecules.model)).asInstanceOf[py.Dynamic]
  }

  private def nodes(environment: Environment[T , P]) =
    environment.getNodes.iterator().asScala.toList
}
