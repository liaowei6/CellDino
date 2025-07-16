from collections import deque
import torch



class Tracker:
    def __init__(self, detection_obj_score_thresh = 0.3, 
        track_obj_score_thresh = 0.2,
        detection_nms_thresh = 0.5,
        track_nms_thresh = 0.15,
        public_detections = None,
        inactive_patience = 5,
        logger = None,
        detection_query_num = None,
        dn_query_num = None,
        track_min_area = 0.0018,
        track_min_distance = 0.11,
        special_frame = [],
        ):
        
        self.detection_obj_score_thresh = detection_obj_score_thresh
        self.track_obj_score_thresh = track_obj_score_thresh
        self.detection_nms_thresh = detection_nms_thresh
        self.track_nms_thresh = track_nms_thresh

        self.public_detections = public_detections
        self.inactive_patience = inactive_patience
        self.detection_query_num = detection_query_num
        self.dn_query_num = dn_query_num
        self._logger = logger
        if self._logger is None:
            self._logger = lambda *log_strs: None
        #training use
        self.track_ids = None
        self.track_index = 0 
        self.track_pos = None
        self.track_query = None
        self.moists = None
        self.dataset=None
        self.dataset_index=None
        self.track_min_area = track_min_area
        self.track_min_distance = track_min_distance
        self.special_frame = [] #仅用于处理测试中部分帧的特殊情况
        #self.reid_sim_threshold = tracker_cfg['reid_sim_threshold']
        #self.reid_sim_only = tracker_cfg['reid_sim_only']
        #self.generate_attention_maps = generate_attention_maps
        #self.reid_score_thresh = tracker_cfg['reid_score_thresh']
        #self.reid_greedy_matching = tracker_cfg['reid_greedy_matching']
        #self.prev_frame_dist = tracker_cfg['prev_frame_dist']
        #self.steps_termination = tracker_cfg['steps_termination']

    def reset(self, hard=True):
        self.tracks = []
        self.inactive_tracks = []
        #self._prev_features = deque([None], maxlen=self.prev_frame_dist)
        self.track_ids = None
        self.moists = None
        self.track_index = 0
        self.track_num = 0
        self.results = {}
        self.frame_index = 0

    def track_to_lost(self, track, frame):
        track.lost(frame)
        #self.inactive_tracks -= track

    def track_to_inactive(self, track, frame):
        track.inactive(frame)
        #self.inactive_tracks += track

    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]

        for track in tracks:
            track.pos = track.last_pos[-1]
        self.inactive_tracks += tracks

    def add_track(self, pos, scores, query_embeds, mather_id=None):
        self.tracks.append(Track(
            query_embeds,
            pos,
            scores,
            self.track_index,
            self.frame_index,
            mather_id,
        )
        )
        self.track_index += 1

    def add_tracks(self, pos, scores, query_embeds, masks=None, attention_maps=None, aux_results=None):
        """Initializes new Track objects and saves them."""
        for i in range(len(pos)):
            self.tracks.append(Track(
                query_embeds[i],
                pos[i],
                scores[i],
                self.track_index + i,
                self.frame_index,
            ))
        self.track_index += len(pos)

    
    def step(self):
    #     """This function should be called every timestep to perform tracking with a blob
    #     containing the image information.
    #     """
    #    inactive_tracks = []
        tracks = []
        track_pos = []
        track_query = []
        track_ids = []
        for track in self.tracks:
            if track.has_positive_area and track.state != 2:
                tracks.append(track)
                track_pos.append(track.pos.unsqueeze(0))
                track_query.append(track.query_emb.unsqueeze(0))
                track_ids.append(track.id)
        self.track_ids = track_ids
        self.track_pos = torch.cat(track_pos, dim=0).unsqueeze(0)
        self.track_query = torch.cat(track_query, dim=0).unsqueeze(0)
        self.tracks = tracks
        self.track_num = len(self.tracks)
        self.frame_index += 1



class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(self, query_emb, pos, score, track_id, start_frame=0, obj_ind=None,
                 mather_id=None, mask=None, attention_map=None):
        self.id = track_id   #轨迹id
        self.query_emb = query_emb  #query
        self.pos = pos #位置编码
        self.last_pos = deque([pos.clone()]) #追踪丢失前最后的位置
        self.score = score #预测分数
        self.ims = deque([])
        self.count_inactive = 0  #丢失帧数
        self.count_termination = 0
        self.gt_id = None #训练时对应的真实样本id
        self.obj_ind = obj_ind
        self.mather_id = mather_id #母细胞id
        self.mask = mask
        self.attention_map = attention_map
        self.state = 3   #0代表inactive, 1代表track, 2表示lost 
        self.start_frame = start_frame
        self.end_frame = None


    def update(self, query_emb, pos, score):
        self.query_emb = query_emb
        self.pos = pos
        self.score = score
        self.state = 1
        self.count_inactive = 0

    def moist(self, query_emb, pos, score, id):
        self.father_id = self.id
        self.query_emb = query_emb
        self.score = score 
        self.pos = pos
        self.id = id 

    def inactive(self, frame):
        self.count_inactive += 1
        self.state = 0
        self.end_frame = frame

    def lost(self, frame):
        self.state = 2
        self.end_frame = frame

    def has_positive_area(self) -> bool:
        """Checks if the current position of the track has
           a valid, .i.e., positive area, bounding box."""
        return self.pos[2] * self.pos[3] > 0.001 

    def reset_last_pos(self) -> None:
        """Reset last_pos to the current position of the track."""
        self.last_pos.clear()
        self.last_pos.append(self.pos.clone())