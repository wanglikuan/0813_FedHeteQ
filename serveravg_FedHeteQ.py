from flcore.clients.clientavg_FedHeteQ import clientAVG
from flcore.servers.serverbase import Server
#from utils.data_utils import read_client_data
from utils.model_utils import read_data, read_user_data, get_public_data_dir, read_public_data #added
from threading import Thread
from flcore.trainmodel.models import *
import time


class FedHeteQ(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, join_clients,
                 num_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, time_threthold):
        super().__init__(dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, join_clients,
                         num_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, 
                         time_threthold)
        # select slow clients
        self.set_slow_clients()
        self.server_public_logits = []
########################################################
        #self.no_Q_model = DigitModel().to('cuda:0')
############################
########################################################

        data = read_data(self.dataset) #added


        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            #train, test = read_client_data(dataset, i)
            id, train , test = read_user_data(i, data, dataset=self.dataset) #added
            client = clientAVG(device, i, train_slow, send_slow, train, test, model, batch_size, learning_rate, local_steps)
            self.clients.append(client)

        print(f"\nJoin clients / total clients: {self.join_clients} / {self.num_clients}")
        print("Finished creating server and clients.")

########################################################
        public_data = read_data('Mnist-alpha20.0-ratio0.5')
        id, train ,used_public_data = read_user_data(18, public_data, dataset=self.dataset)
        self.selected_clients = self.select_clients()
        for client in self.selected_clients:
            client.get_publice_data(used_public_data)
############################
########################################################

    def test_accuracy_Q(self):
        num_samples = []
        tot_correct = []
        for c in self.clients:
            ct, ns = c.test_accuracy_Q()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct

    def train_accuracy_and_loss_Q(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.clients:
            ct, cl, ns = c.train_accuracy_and_loss_Q()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, losses

    # evaluate all clients
    def evaluate_Q(self):
        stats = self.test_accuracy_Q()
        stats_train = self.train_accuracy_and_loss_Q()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        train_acc = sum(stats_train[2])*1.0 / sum(stats_train[1])
        train_loss = sum(stats_train[3])*1.0 / sum(stats_train[1])
        
        self.rs_test_acc.append(test_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        self.print_(test_acc, train_acc, train_loss)

        for x,y in zip(stats[2],stats[1]):
            #print("------------------------------")
            print("client Accurancy: ", x*1.0/y)    

    def train(self):
        for i in range(self.global_rounds+1):
            #self.send_models()
            #self.send_parameters_fedrep()
            #self.send_parameters_fedbn()


            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate_Q()

            self.timestamp = time.time() #
            self.selected_clients = self.select_clients()
            for client in self.selected_clients:
                client.train()
            curr_timestamp = time.time() #
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_clients)
            print("glob_iter: ",i,"    train_time: ",train_time)

            #########得到每个client的logits#############
            for client in self.selected_clients:
                if client.id == 0:
                    client.predict()
                    self.server_public_logits = client.get_truelabel()
                    print("got client 0 logits")
                else:
                    client.predict()
            ######################

            #########将public logits下发给client#############
            for client in self.selected_clients:
                client.get_public_logits(self.server_public_logits)
            ######################

            #########client 进行对Q的知识蒸馏#############
            for client in self.selected_clients:
                client.train_Q()
            ######################

            self.timestamp = time.time() #
            #self.receive_models()
            #self.aggregate_parameters()
            curr_timestamp = time.time() #
            agg_time = curr_timestamp - self.timestamp
            print("glob_iter: ",i,"    agg_time: ",agg_time)

        for client in self.selected_clients:
            client.savemodel()

        print("\nBest global results.")
        self.print_(max(self.rs_test_acc), max(
            self.rs_train_acc), min(self.rs_train_loss))

        self.save_results()
        self.save_global_model()
