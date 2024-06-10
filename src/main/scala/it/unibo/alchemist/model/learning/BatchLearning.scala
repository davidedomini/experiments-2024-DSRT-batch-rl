package it.unibo.alchemist.model.learning

import it.unibo.alchemist.boundary.launchers.LearningLauncher
import scala.jdk.CollectionConverters.SeqHasAsJava
import it.unibo.alchemist.boundary.LoadAlchemist

case class BatchLearning (
    strategies: List[ExecutionStrategy[Any, Nothing]],
    simulationConfiguration: String,
    batch: List[String],
    globalRounds: Int,
    miniBatchSize: Int,
    seedName: String
  ){

  def startLearning(): Unit = {
    val loader = LoadAlchemist.from(simulationConfiguration)
    val launcher = new LearningLauncher(
      new java.util.ArrayList(batch.asJava),
      false,
      false,
      globalRounds,
      seedName,
      miniBatchSize,
      strategies
    )
    launcher.launch(loader)
  }

}
