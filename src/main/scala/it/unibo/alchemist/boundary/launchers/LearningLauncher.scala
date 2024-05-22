package it.unibo.alchemist.boundary.launchers

import it.unibo.alchemist.boundary.{Launcher, Loader}

class LearningLauncher (
                         val batch: List[String] = List.empty,
                         val autoStart: Boolean = true,
                         val showProgress: Boolean = true,
                         val parallelism: Int = Runtime.getRuntime.availableProcessors(),
                         val globalRounds: Int = 20 // TODO - check
                       ) extends Launcher {

  override def launch(loader: Loader): Unit = ???
}
