"""
Base callback class for compatibility with keras-rl callbacks.
This provides a minimal implementation to replace rl.callbacks.Callback
if keras-rl is not available.
"""

class Callback:
    """
    Base callback class compatible with keras-rl callbacks API.
    """
    def __init__(self):
        pass
    
    def on_episode_begin(self, episode, logs={}):
        """Called at the beginning of an episode."""
        pass
    
    def on_episode_end(self, episode, logs={}):
        """Called at the end of an episode."""
        pass
    
    def on_step_begin(self, step, logs={}):
        """Called at the beginning of a step."""
        pass
    
    def on_step_end(self, step, logs={}):
        """Called at the end of a step."""
        pass
    
    def on_action_begin(self, action, logs={}):
        """Called at the beginning of an action."""
        pass
    
    def on_action_end(self, action, logs={}):
        """Called at the end of an action."""
        pass
