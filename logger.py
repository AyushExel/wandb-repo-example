from pathlib import Path

import wandb

class WandbLogger():
    def __init__(self, **kwargs):
        self.run = wandb.init(**kwargs)
        self.log_dict = {}


    def log(self, log_dict, flush=False):
        '''
        This function accumulates data in self.log_dict. If flush is set.
        the accumulated metrics will be logged directly to wandb dashboard.
        '''
        for key, value in log_dict.items():
            self.log_dict[key] = value
        if flush: # for cases where you don't want to accumulate data
            self.flush()
	
    def flush(self):
        '''
        This function logs the accumulated data to wandb dashboard. This
        function when called once per epoch, logs data for that epoch/step
        '''
        wandb.log(self.log_dict)
        self.log_dict = {}


    def finish(self):
        '''
        Finish this W&B run
        '''
        self.run.finish()

    def log_artifact(self, file_or_dir, name, type, aliases=[]):
        artifact = wandb.Artifact(name, type=type)
        data_path = Path(file_or_dir)
        if data_path.is_dir():
            artifact.add_dir(str(data_path))
        elif data_path.is_file():
            artifact.add_file(str(data_path))

        self.run.log_artifact(artifact, aliases=aliases )
        


		
