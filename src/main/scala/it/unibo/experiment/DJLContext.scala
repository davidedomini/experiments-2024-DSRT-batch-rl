package it.unibo.experiment

import ai.djl.ndarray.NDManager

object DJLContext {
  val globalManager = NDManager.newBaseManager()

  def localManager() = globalManager.newSubManager()
}
