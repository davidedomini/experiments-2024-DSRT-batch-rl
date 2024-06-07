package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.{Environment, Position, TimeDistribution}
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.learning.Molecules
import it.unibo.experiment.SimpleSequentialDQN
import me.shadaj.scalapy.py.SeqConverters
import it.unibo.interop.PythonModules._
import me.shadaj.scalapy.py

import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters.IteratorHasAsScala

class ActionChoiceReaction[T, P <: Position[P]] (
    environment: Environment[T, P],
    distribution: TimeDistribution[T]
  ) extends AbstractGlobalReaction(environment, distribution) {

  private val seed = 0.0

  override protected def executeBeforeUpdateDistribution(): Unit = {

    val policy = loadNN()

    val observations = nodes
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

  private def loadNN(): py.Dynamic = {
    nodes.head.getConcentration(new SimpleMolecule(Molecules.model)).asInstanceOf[py.Dynamic]
  }

}