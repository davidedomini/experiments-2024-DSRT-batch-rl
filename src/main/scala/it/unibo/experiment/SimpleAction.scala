package it.unibo.experiment

import ai.djl.engine.Engine
import ai.djl.{Device, Model}
import ai.djl.ndarray.types.{DataType, Shape}
import ai.djl.ndarray.{BaseNDManager, NDArray, NDArrays, NDList, NDManager}
import ai.djl.nn.{Activation, SequentialBlock}
import ai.djl.nn.core.Linear
import ai.djl.translate.{NoopTranslator, Translator}
import it.unibo.alchemist.model.{Action, Context, Node, Reaction}
import it.unibo.alchemist.model.actions.AbstractAction
import it.unibo.experiment.SimpleAction.blocks

import scala.::
import scala.jdk.CollectionConverters.CollectionHasAsScala

class SimpleAction[T](node: Node[T]) extends AbstractAction[T](node) {
  override def cloneAction(node: Node[T], reaction: Reaction[T]): Action[T] = new SimpleAction(node)

  override def execute(): Unit = {
    import SimpleAction._
    val localManager = NDManager.newBaseManager(Device.gpu()).asInstanceOf[BaseNDManager]
    val model = Model.newInstance("local")
    model.setBlock(blocks)
    val predictor = model.newPredictor(new NoopTranslator(), Device.gpu())
    val input = new NDList(localManager.randomNormal(new Shape(1024)))
    predictor.predict(input)
    model.close()
    localManager.close()
  }

  override def getContext: Context = Context.LOCAL
}

object SimpleAction {
  val blocks = new SequentialBlock()
    .add(Linear.builder().setUnits(1024).build())
    .add(Activation.reluBlock())
    .add(Linear.builder().setUnits(256).build())
    .add(Activation.reluBlock())
    .add(Linear.builder().setUnits(8).build())

  val tensorManager: BaseNDManager = NDManager.newBaseManager(Device.gpu()).asInstanceOf[BaseNDManager]
  blocks.initialize(tensorManager, DataType.FLOAT32, new Shape(1024))
}
