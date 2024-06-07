package it.unibo.experiment

import it.unibo.interop.PythonModules.torch
import me.shadaj.scalapy.py

object SimpleSequentialDQN {
  def apply(input: Int, hidden: Int, output: Int): py.Dynamic =
    torch.nn
      .Sequential(
        torch.nn.Linear(input, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, output)
      )
}