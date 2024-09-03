from metaflow import FlowSpec, step, Parameter, Flow


class MasterFlow(FlowSpec):

    @step
    def start(self):
        print("Starting the master flow")
        self.next(self.run_preprocessing)

    @step
    def run_preprocessing(self):
        preprocessing_flow = Flow('PreprocessingFlow')
        preprocessing_flow.run()

        self.preprocessed_train = preprocessing_flow.data.reduced_train
        self.preprocessed_test = preprocessing_flow.data.reduced_test
        self.labels = preprocessing_flow.data.labels
        
        print("Preprocessing completed.")
        self.next(self.run_keras_tuner)

    @step
    def run_keras_tuner(self):
        keras_tuner_flow = Flow('KerasTunerFlow')
        keras_tuner_flow.run(parameters={
            'X': self.preprocessed_train,
            'y': self.labels
        })

        print("KerasTuner completed.")
        self.next(self.end)

    @step
    def end(self):
        print("Master flow completed successfully.")
