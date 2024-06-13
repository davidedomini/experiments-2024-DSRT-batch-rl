package it.unibo.experiment

import it.unibo.interop.PythonModules.nn
import me.shadaj.scalapy.py

object SimpleSequentialDQN {
  def apply(input: Int, hidden: Int, output: Int): py.Dynamic =
    nn
      .Sequential(
        nn.Linear(input, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, output)
      )
}
