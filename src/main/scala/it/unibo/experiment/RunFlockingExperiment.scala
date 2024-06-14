package it.unibo.experiment

import it.unibo.alchemist.boundary.launchers.DeepQLearningLauncher
import it.unibo.alchemist.model.learning.{BatchLearning, LearningLauncher}

import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters.IterableHasAsJava

object RunFlockingExperiment extends App {

  private def strategies = List(
    new StateEvaluationStrategy[Any, Nothing](true),
    new ActionChoiceStrategy[Any, Nothing](initialEpsilon = 0.9, decayFactor = 0.1),
    new CollectiveActionExecutionStrategy[Any, Nothing],
    new StateEvaluationStrategy[Any, Nothing](false),
    new RewardEvaluationStrategy[Any, Nothing](ExperimentParams.minDistance, ExperimentParams.maxDistance),
    new ExperienceCollectionStrategy[Any, Nothing]
  )

  def scenarioWith(file: String, parallelism: Int): Unit = {
    val learner = new DeepQLearningLauncher(
      batch = new java.util.ArrayList(java.util.List.of("seed")),
      globalRounds = 10,
      parallelism = parallelism,
      seedName = "seed",
      strategies = strategies,
      globalSeed = 42,
      globalBufferSize = 4000000,
      learningInfo = DeepQLearningLauncher.LearningInfo(gamma = 0.9),
      networkFactory = new DeepQLearningLauncher.DQNFactory(
        ExperimentParams.neighbors * ExperimentParams.neighborPositionSize,
        128,
        ActionSpace.all.size
      )
    )
    LearningLauncher(file, learner)
  }
  Files.write(Paths.get("computingTime.csv"), "parallelism,computingTime\n".getBytes)
  private val parallel = List(1, 2, 4, 8)
  parallel.foreach { p =>
    val time = System.currentTimeMillis()
    scenarioWith(s"src/main/yaml/simulation$p.yml", p)
    val computingTime = System.currentTimeMillis() - time
    val content = s"$p,$computingTime\n"
    Files.write(Paths.get("computingTime.csv"), content.getBytes, java.nio.file.StandardOpenOption.APPEND)
  }
}
