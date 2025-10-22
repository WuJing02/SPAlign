import copy
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.modules import loss
from tqdm import tqdm

import pcode.create_aggregator as create_aggregator
import pcode.create_dataset as create_dataset
import pcode.create_metrics as create_metrics
import pcode.create_model as create_model
import pcode.create_optimizer as create_optimizer
import pcode.create_scheduler as create_scheduler
import pcode.local_training.random_reinit as random_reinit
import pcode.models as models
from pcode import master_utils
from pcode.utils.logging import display_training_stat
from pcode.utils.stat_tracker import RuntimeTracker
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.workers.worker_base import WorkerBase

class WorkerSPAlign(WorkerBase):
    def __init__(self, conf):
        super().__init__(conf)
        self.M = 1
        self.running_mean = None
        self.running_var = None
        self.alpha = 0.7

    def run(self):
        while True:
            self._listen_to_master()
            if self._terminate_by_early_stopping():
                return
            self._recv_model_from_master()
            if self.is_active_before == 0:
                self._train()
            else:
                self._train_AKT()
            self._send_model_to_master()
            if self._terminate_by_complete_training():
                return

    def _listen_to_master(self):
        msg = torch.zeros((4, self.conf.n_participated))
        dist.broadcast(tensor=msg, src=0)
        self.conf.graph.client_id, self.conf.graph.comm_round, self.n_local_epochs, self.is_active_before = (
            msg[:, self.conf.graph.rank - 1].to(int).cpu().numpy().tolist()
        )
        self.arch, self.model = create_model.define_model(
            self.conf, to_consistent_model=False, client_id=self.conf.graph.client_id
        )
        self.model_state_dict = self.model.state_dict()
        self.model_tb = TensorBuffer(list(self.model_state_dict.values()))
        self.metrics = create_metrics.Metrics(self.model, task="classification")
        dist.barrier()

    def _recv_model_from_master(self):
        dist.recv(self.model_tb.buffer, src=0)
        self.model_tb.unpack(self.model_state_dict.values())
        self.model.load_state_dict(self.model_state_dict)
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) received the global/personal model ({self.arch}) from Master."
        )
        if self.is_active_before == 1:
            flatten_local_models = []
            for i in range(self.M):
                client_tb = TensorBuffer(
                    list(copy.deepcopy(self.model.state_dict()).values())
                )
                client_tb.buffer = torch.zeros_like(client_tb.buffer)
                flatten_local_models.append(client_tb)
            for i in range(self.M):
                dist.recv(tensor=flatten_local_models[i].buffer, src=0)
            self.last_local_model = copy.deepcopy(self.model)
            _last_model_state_dict = self.last_local_model.state_dict()
            flatten_local_models[0].unpack(_last_model_state_dict.values())
            self.last_local_model.load_state_dict(_last_model_state_dict)
            self.conf.logger.log(
                f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) received ({self.M}) past local models from Master."
            )
        dist.barrier()

    def _train_AKT(self):
        self.model.train()
        self.prepare_local_train_loader()
        if self.conf.graph.on_cuda:
            self.model = self.model.cuda()
            self.last_local_model = self.last_local_model.cuda()
        self.optimizer = create_optimizer.define_optimizer(
            self.conf, model=self.model, optimizer_name=self.conf.optimizer
        )
        self.scheduler = create_scheduler.Scheduler(self.conf, optimizer=self.optimizer)
        self.tracker = RuntimeTracker(metrics_to_track=self.metrics.metric_names)
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) enters the local training phase (current communication rounds={self.conf.graph.comm_round})."
        )
        while True:
            for _input, _target in self.train_loader:
                with self.timer("load_data", epoch=self.scheduler.epoch_):
                    data_batch = create_dataset.load_data_batch(
                        self.conf, _input, _target, is_training=True
                    )
                with self.timer("forward_pass", epoch=self.scheduler.epoch_):
                    self.optimizer.zero_grad()
                    loss, _ = self._local_training_with_last_local_model(data_batch)
                with self.timer("backward_pass", epoch=self.scheduler.epoch_):
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                if (
                    self.conf.display_tracked_time
                    and self.scheduler.local_index % self.conf.summary_freq == 0
                ):
                    self.conf.logger.log(self.timer.summary())
                if self.tracker.stat["loss"].avg > 1e3 or np.isnan(self.tracker.stat["loss"].avg):
                    self.conf.logger.log(
                        f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) diverges!!!!!Early stop it."
                    )
                    self._terminate_comm_round()
                    return
                if self._is_finished_one_comm_round():
                    display_training_stat(self.conf, self.scheduler, self.tracker)
                    self._terminate_comm_round()
                    return
            display_training_stat(self.conf, self.scheduler, self.tracker)
            self.tracker.reset()
            if self.conf.logger.meet_cache_limit():
                self.conf.logger.save_json()

    def _perception_reconstruction(self, logits):
        batch_size, num_classes = logits.size()
        mean_per_class = logits.mean(dim=0, keepdim=True)
        var_per_class = logits.var(dim=0, keepdim=True, unbiased=False)

        if self.running_mean is None:
            self.running_mean = mean_per_class.detach().clone()
            self.running_var = var_per_class.detach().clone()
        else:
            self.running_mean = (self.alpha * self.running_mean + (1 - self.alpha) * mean_per_class).detach()
            self.running_var = (self.alpha * self.running_var + (1 - self.alpha) * var_per_class).detach()

        eps = 1e-8
        var_per_class = self.running_var.clamp(min=eps)
        perception_logits = (logits - self.running_mean) / torch.sqrt(var_per_class)
        return perception_logits

    def _local_training_with_last_local_model(self, data_batch):
        loss, output = self._inference(data_batch)
        performance = self.metrics.evaluate(loss, output, data_batch["target"])
        last_local_logit = self.last_local_model(data_batch["input"])

        student_perception_logits = self._perception_reconstruction(output)
        teacher_perception_logits = self._perception_reconstruction(last_local_logit)

        loss2 = self.conf.lamda * self._divergence(
            student_logits=student_perception_logits,
            teacher_logits=teacher_perception_logits,
            KL_temperature=self.conf.KL_T,
        )

        loss_all = loss + loss2
        if self.tracker is not None:
            self.tracker.update_metrics(
                [loss.item()] + performance + [loss2.item()], n_samples=data_batch["input"].size(0)
            )
        return loss_all, output

    def _terminate_comm_round(self):
        self.model = self.model.cpu()
        if hasattr(self, 'init_model'):
            del self.init_model
        self.scheduler.clean()
        self.conf.logger.save_json()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) finished one round of federated learning: (comm_round={self.conf.graph.comm_round})."
        )

    def _divergence(self, student_logits, teacher_logits, KL_temperature):
        return F.kl_div(
            F.log_softmax(student_logits / KL_temperature, dim=1),
            F.softmax(teacher_logits / KL_temperature, dim=1),
            reduction="batchmean"
        ) * (KL_temperature ** 2)

def sigmoid_rampup(current, rampup_length=3):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def linear_rampup(current, rampup_length=15):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

def attention(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def attention_diff(x, y):
    return (attention(x) - attention(y)).pow(2).mean()