package it.unibo.experiment

import ai.djl.Model
import ai.djl.ndarray.NDList
import ai.djl.nn.Block
import ai.djl.translate.NoopTranslator
import it.unibo.alchemist.model.learning.{GlobalExecution, Molecules}
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{Environment, Position}
import org.apache.commons.math3.random.RandomGenerator

import scala.jdk.CollectionConverters.IteratorHasAsScala

class ActionChoiceStrategy[T, P <: Position[P]](initialEpsilon: Double, decayFactor: Double)
    extends GlobalExecution[T, P] {
  var epsilon = initialEpsilon

  override def progressRound(): Unit = {
    epsilon -= decayFactor
    super.progressRound()
  }
  override def execute(environment: Environment[T, P], random: RandomGenerator): Unit = {
    val policy = loadNN(environment)
    var actions: List[Int] = List.empty
    val r = random.nextDouble()
    val local = DJLContext.localManager()
    if (r < epsilon) {
      actions = nodes(environment).map(n => random.nextInt(ActionSpace.all.size))
    } else {
      val observations = nodes(environment)
        .map { node =>
          node.getConcentration(new SimpleMolecule(Molecules.encodedActualState)).asInstanceOf[List[Double]]
        }
      val observationsTensor = local.create(observations.map(_.toArray).toArray)
      val model = Model.newInstance("inference")
      model.setBlock(policy)
      val predictor = model.newPredictor(new NoopTranslator())
      val qValues = predictor.predict(new NDList(observationsTensor))
      actions = qValues.head().argMax(1).toLongArray.toList.map(_.toInt)
      model.close()
      predictor.close()
      local.close()
    }
    actions.zipWithIndex
      .foreach { case (action, index) =>
        environment.getNodeByID(index).setConcentration(new SimpleMolecule(Molecules.action), action.asInstanceOf[T])
      }
  }

  private def loadNN(environment: Environment[T, P]): Block = {
    val position = environment.getPosition(nodes(environment).head)
    environment.getLayer(new SimpleMolecule(Molecules.model)).get().getValue(position).asInstanceOf[Block]
  }

  private def nodes(environment: Environment[T, P]) =
    environment.getNodes.iterator().asScala.toList
}
