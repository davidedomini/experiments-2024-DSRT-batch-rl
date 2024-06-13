package it.unibo.alchemist.model.learning

import it.unibo.alchemist.boundary.launchers.LearningLauncher
import scala.jdk.CollectionConverters.SeqHasAsJava
import it.unibo.alchemist.boundary.LoadAlchemist

case class BatchLearning(
    strategies: List[ExecutionStrategy[Any, Nothing]],
    simulationConfiguration: String,
    batch: List[String],
    globalRounds: Int,
    parallelism: Int,
    miniBatchSize: Int,
    seedName: String
) {

  def startLearning(): Unit = {
    val loader = LoadAlchemist.from(simulationConfiguration)
    val launcher = new LearningLauncher(
      batch = new java.util.ArrayList(batch.asJava),
      autoStart = false,
      showProgress = false,
      globalRounds = globalRounds,
      parallelism = parallelism,
      seedName = seedName,
      miniBatchSize = miniBatchSize,
      strategies = strategies
    )
    launcher.launch(loader)
  }

}
