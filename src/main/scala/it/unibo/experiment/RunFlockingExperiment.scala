package it.unibo.experiment

import it.unibo.alchemist.boundary.launchers.DeepQLearningLauncher
import it.unibo.alchemist.model.learning.{BatchLearning, LearningLauncher}

import scala.jdk.CollectionConverters.IterableHasAsJava

object RunFlockingExperiment extends App {

  private def strategies = List(
    new StateEvaluationStrategy[Any, Nothing](true),
    new ActionChoiceStrategy[Any, Nothing](initialEpsilon = 0.9, decayFactor = 0.1),
    new CollectiveActionExecutionStrategy[Any, Nothing],
    new StateEvaluationStrategy[Any, Nothing](false),
    new RewardEvaluationStrategy[Any, Nothing],
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
      learningInfo = DeepQLearningLauncher.LearningInfo(),
      networkFactory = new DeepQLearningLauncher.DQNFactory(10, 256, ActionSpace.all.size)
    )
    LearningLauncher(file, learner)
  }

  val time = System.currentTimeMillis()
  scenarioWith("src/main/yaml/simulation.yml", 1)
  val computingTime = System.currentTimeMillis() - time

  val time2 = System.currentTimeMillis()
  scenarioWith("src/main/yaml/simulation2.yml", 2)
  val computingTime2 = System.currentTimeMillis() - time2

  val time4 = System.currentTimeMillis()
  scenarioWith("src/main/yaml/simulation4.yml", 4)
  val computingTime4 = System.currentTimeMillis() - time4

  val content = s"1,$computingTime\n2,$computingTime2\n4,$computingTime4"

  println(content)
}
