from torch.utils.data import Dataset
import utils.openbias_utils as utils
from collections import defaultdict
import json, os, sys
from PIL import Image

#####################################################################################
#######                                                                       #######
#######                     IMAGE GENERATION DATASETS                         #######
#######                                                                       #######
#####################################################################################
class Proposed_biases(Dataset):
    def __init__(
        self,
        dataset_path,
        max_prompts,
        filter_threshold,
        hard_threshold,
        merge_threshold,
        valid_bias_fn,
        filter_caption_fn,
        all_images,
        specifc_biases=None
    ):
        super(Proposed_biases).__init__()
        self.max_prompts = max_prompts

        print("Loading and filtering proposed biases...")
        # post process LLM output
        captions, image_ids, bias_classes_final, bias_captions_final, class_clusters_merged, class_clusters_string_merged = utils.post_processing(
            dataset_path,
            threshold=filter_threshold,
            hard_threshold=hard_threshold,
            merge_threshold=merge_threshold,
            valid_bias_fn=valid_bias_fn,
            filter_caption_fn=filter_caption_fn,
            all_images=all_images
        )
        print("Done!")
        # prompts to use for image generation 
        # we take one <max_prompts> 
        self.prompts = set()
        if specifc_biases is None:
            # for each bias group
            for bias_group_name in bias_captions_final:
                # for each bias
                for bias_name in bias_captions_final[bias_group_name]:
                    # for each class cluster
                    for class_cluster in bias_captions_final[bias_group_name][bias_name]:
                        # get first caption captions
                        captions_ids = utils.get_first_caption(
                            captions_id = bias_captions_final[bias_group_name][bias_name][class_cluster],
                            captions = captions,
                            max_prompts = max_prompts
                        )
                        # for each caption
                        for caption_id, question in captions_ids:
                            # get caption and image id
                            caption, image_id = captions[caption_id]
                            # add prompt
                            self.prompts.add((bias_cluster, caption, caption_id))
            self.bias_captions_final = bias_captions_final
            self.bias_classes_final = bias_classes_final
        else:
            self.bias_captions_final = {}
            self.bias_classes_final = {}
            for bias_cluster, bias_name in specifc_biases:
                cluster = list(bias_captions_final[bias_cluster][bias_name].keys())[0]
                if bias_cluster not in self.bias_captions_final:
                    self.bias_captions_final[bias_cluster] = {}
                    self.bias_classes_final[bias_cluster] = {}
                if bias_name not in self.bias_captions_final[bias_cluster]:
                    self.bias_captions_final[bias_cluster][bias_name] = {cluster: []}
                    self.bias_classes_final[bias_cluster][bias_name] = {
                        cluster: bias_classes_final[bias_cluster][bias_name][cluster]
                    }
                captions_ids = utils.get_first_caption(
                    captions_id = bias_captions_final[bias_cluster][bias_name][cluster],
                    captions = captions,
                    max_prompts = max_prompts
                )
                for caption_id, question in captions_ids:
                    caption, image_id = captions[caption_id]
                    self.prompts.add((bias_cluster, caption, caption_id))
                    self.bias_captions_final[bias_cluster][bias_name][cluster].append(
                        (
                            caption_id,
                            question
                        )
                    )

        self.prompts = list(self.prompts)
        self.prompts = sorted(self.prompts, key=lambda x: x[1])
        self.captions = captions

    def __len__(self):
        return len(self.prompts)

    def get_bias_captions_id(self):
        return self.bias_captions_id

    def get_bias_classes(self):
        return self.bias_classes_final

    def get_biases(self):
        return self.bias_captions_final, self.bias_classes_final, self.captions

    def get_data(self):
        return self.prompts

    def __getitem__(self, index):
        bias_cluster, caption, caption_id = self.prompts[index]
        return bias_cluster, caption, caption_id

class GradBias_proposed_biases(Dataset):
    def __init__(
        self,
        dataset_path,
        max_prompts,
        filter_threshold,
        hard_threshold,
        merge_threshold,
        valid_bias_fn,
        filter_caption_fn,
        all_images,
        specifc_biases=None
    ):
        super(GradBias_proposed_biases).__init__()
        self.max_prompts = max_prompts

        print("Loading and filtering proposed biases...")
        # post process LLM output
        captions, image_ids, bias_classes_final, bias_captions_final, class_clusters_merged, class_clusters_string_merged = utils.post_processing(
            dataset_path,
            threshold=filter_threshold,
            hard_threshold=hard_threshold,
            merge_threshold=merge_threshold,
            valid_bias_fn=valid_bias_fn,
            filter_caption_fn=filter_caption_fn,
            all_images=all_images
        )
        print("Done!")
        # prompts to use for image generation 
        # we take one <max_prompts> 
        self.prompts = set()
        if specifc_biases is None:
            # for each bias group
            for bias_group_name in bias_captions_final:
                # for each bias
                for bias_name in bias_captions_final[bias_group_name]:
                    # for each class cluster
                    for class_cluster in bias_captions_final[bias_group_name][bias_name]:
                        # get first caption captions
                        captions_ids = utils.get_first_caption(
                            captions_id = bias_captions_final[bias_group_name][bias_name][class_cluster],
                            captions = captions,
                            max_prompts = max_prompts
                        )
                        # for each caption
                        for caption_id, question in captions_ids:
                            # get caption and image id
                            caption, image_id = captions[caption_id]
                            # add prompt
                            self.prompts.add(
                                (
                                    caption, 
                                    caption_id,
                                    bias_group_name,
                                    bias_name,
                                    class_cluster,
                                    question
                                )
                            )
            self.bias_captions_final = bias_captions_final
            self.bias_classes_final = bias_classes_final
        else:
            for bias_cluster, bias_name in specifc_biases:
                cluster = list(bias_captions_final[bias_cluster][bias_name].keys())[0]
                captions_ids = utils.get_first_caption(
                    captions_id = bias_captions_final[bias_cluster][bias_name][cluster],
                    captions = captions,
                    max_prompts = max_prompts
                )

                for caption_id, question in captions_ids:
                    caption, image_id = captions[caption_id]
                    self.prompts.add(
                        (
                            caption, 
                            caption_id,
                            bias_cluster,
                            bias_name,
                            cluster,
                            question,
                        )
                    )
            self.bias_classes_final = bias_classes_final

        self.prompts = list(self.prompts)
        self.prompts = sorted(self.prompts, key=lambda x: x[1])
        self.captions = captions

    def __len__(self):
        return len(self.prompts)

    def get_data(self):
        return self.prompts

    def get_classes(self):
        return self.bias_classes_final

    def __getitem__(self, index):
        caption, caption_id = self.prompts[index]
        return caption, caption_id

class VQA_dataset(Dataset):
    def __init__(
        self,
        dataset_setting,
        save_path,
        file_name,
        multi_image=False
    ):
        super().__init__()
        self.multi_image = multi_image
        images_paths = dataset_setting['images_path']
        proposed_biases_path = dataset_setting['path']
        # get predicted biases
        bias_dataset = Proposed_biases(
            dataset_path = proposed_biases_path,
            max_prompts = dataset_setting['max_prompts_per_bias'],
            filter_threshold = dataset_setting['filter_threshold'],
            hard_threshold = dataset_setting['hard_threshold'],
            merge_threshold = dataset_setting['merge_threshold'],
            valid_bias_fn = dataset_setting['valid_bias_fn'],
            filter_caption_fn = dataset_setting['filter_caption_fn'],
            all_images = dataset_setting['all_images'],
            specifc_biases = dataset_setting['biases_to_check']
        )

        captions_done = set()
        if not self.multi_image:
            if save_path is not None and file_name is not None and os.path.exists(os.path.join(save_path, file_name)):
                with open(os.path.join(save_path, file_name), 'r') as f:
                    vqa_answers = json.load(f)
                for image_name in vqa_answers:
                    caption_id = image_name.split('/')[-3]
                    tested_word = image_name.split('/')[-2]
                    image_id = image_name.split('/')[-1]
                    captions_done.add(f'{caption_id}/{tested_word}/{image_id}')
        else:
            if save_path is not None and file_name is not None and os.path.exists(os.path.join(save_path, file_name)):
                with open(os.path.join(save_path, file_name), 'r') as f:
                    vqa_answers = json.load(f)
                for caption_id in vqa_answers:
                    captions_done.add(f'{caption_id}')

        # get biases and captions
        bias_captions_final, bias_classes_final, captions = bias_dataset.get_biases()

        # define dict of biases
        # biases = {
        #     'caption_id': [
        #         (
        #            bias_cluster,
        #            bias_name,
        #            classes_cluster,
        #            question,
        #            [classes]
        #         ),
        #        ...    
        #     ]
        # }
        biases = defaultdict(list)
        # for each bias cluster
        for bias_cluster in bias_captions_final:
            # for each bias
            for bias_name in bias_captions_final[bias_cluster]:
                # for each class cluster
                for class_cluster in bias_captions_final[bias_cluster][bias_name]:
                    # get first caption for each real image
                    cpts = utils.get_first_caption(
                        captions_id = bias_captions_final[bias_cluster][bias_name][class_cluster],
                        captions = captions,
                        max_prompts = dataset_setting['max_prompts_per_bias']
                    )
                    # for each caption
                    for cpt_id, question in cpts:
                        # add bias information to the dict of caption ids
                        biases[cpt_id].append(
                            (
                                bias_cluster,
                                bias_name,
                                class_cluster,
                                question,
                                bias_classes_final[bias_cluster][bias_name][class_cluster]['classes']
                            )
                        )

        # define data
        # data = [
        #     (
        #         caption_id,
        #         caption,
        #         image_path,
        #         [
        #             (
        #                 bias_cluster,
        #                 bias_name,
        #                 class_cluster,
        #                 question,
        #                 [classes]
        #             ),
        #             ...
        #         ]
        #     ),
        #     ...
        # ]
        self.data = []
        # for each caption id
        for caption_id in biases:
            # for each tested word
            if os.path.exists(os.path.join(images_paths, str(caption_id))):
                tested_words = os.listdir(os.path.join(images_paths, str(caption_id)))
                if not multi_image:
                    for word in tested_words:
                        # get images path
                        image_path = os.path.join(images_paths, str(caption_id), word)
                        # get list of images
                        images = os.listdir(image_path)
                        # for each image
                        for image_name in images:
                            if f'{caption_id}/{word}/{image_name}' not in captions_done:
                                # save bias information regarding the image
                                self.data.append(
                                    (
                                        caption_id, 
                                        captions[caption_id][0],
                                        captions[caption_id][1],
                                        os.path.join(image_path, image_name), 
                                        biases[caption_id]
                                    )
                                )
                else:
                    candidate_words = [word for word in tested_words if word != 'full_prompt']
                    # get images path
                    image_path = os.path.join(images_paths, str(caption_id))
                    if f'{caption_id}' not in captions_done:
                        self.data.append(
                            (
                                caption_id, 
                                captions[caption_id][0],
                                captions[caption_id][1],
                                image_path, 
                                candidate_words,
                                biases[caption_id]
                            )
                        )
        self.bias_classes_final = bias_classes_final
        
    def __len__(self):
        return len(self.data)

    def get_data(self):
        return self.data
    
    def get_bias_classes(self):
        return self.bias_classes_final

    def __getitem__(self, index):
        if not self.multi_image:
            # get image info
            caption_id, caption, image_id, image_path, proposed_biases = self.data[index]
            image = Image.open(image_path)
            return caption_id, caption, image_id, image_path, image, proposed_biases
        else:
            # get image info
            caption_id, caption, image_id, general_image_path, candidate_words, proposed_biases = self.data[index]
            loaded_images = {}
            for word in candidate_words:
                image_path = os.path.join(general_image_path, word)
                image_names = os.listdir(image_path)
                for image_name in image_names:
                    loaded_images[os.path.join(image_path, image_name)] = Image.open(os.path.join(image_path, image_name))
            return caption_id, caption, image_id, general_image_path, loaded_images, candidate_words, proposed_biases

class LLM_dataset(Dataset):
    def __init__(
        self,
        dataset_setting,
        save_path,
        file_name
    ):
        super().__init__()
        proposed_biases_path = dataset_setting['path']
        # get predicted biases
        bias_dataset = Proposed_biases(
            dataset_path = proposed_biases_path,
            max_prompts = dataset_setting['max_prompts_per_bias'],
            filter_threshold = dataset_setting['filter_threshold'],
            hard_threshold = dataset_setting['hard_threshold'],
            merge_threshold = dataset_setting['merge_threshold'],
            valid_bias_fn = dataset_setting['valid_bias_fn'],
            filter_caption_fn = dataset_setting['filter_caption_fn'],
            all_images = dataset_setting['all_images'],
            specifc_biases = dataset_setting['biases_to_check']
        )

        captions_done = set()
        if save_path is not None and file_name is not None and os.path.exists(os.path.join(save_path, file_name)):
            with open(os.path.join(save_path, file_name), 'r') as f:
                LLM_answers = json.load(f)
            captions_done = set(list(LLM_answers.keys()))

        # get biases and captions
        bias_captions_final, bias_classes_final, captions = bias_dataset.get_biases()

        # define dict of biases
        # biases = {
        #     'caption_id': [
        #         (
        #            bias_cluster,
        #            bias_name,
        #            classes_cluster,
        #            question,
        #            [classes]
        #         ),
        #        ...    
        #     ]
        # }
        biases = defaultdict(list)
        # for each bias cluster
        for bias_cluster in bias_captions_final:
            # for each bias
            for bias_name in bias_captions_final[bias_cluster]:
                # for each class cluster
                for class_cluster in bias_captions_final[bias_cluster][bias_name]:
                    # get first caption for each real image
                    cpts = utils.get_first_caption(
                        captions_id = bias_captions_final[bias_cluster][bias_name][class_cluster],
                        captions = captions,
                        max_prompts = dataset_setting['max_prompts_per_bias']
                    )
                    # for each caption
                    for cpt_id, question in cpts:
                        if str(cpt_id) not in captions_done:
                            # add bias information to the dict of caption ids
                            biases[cpt_id].append(
                                (
                                    bias_cluster,
                                    bias_name,
                                    class_cluster
                                )
                            )
        
        self.data = []
        for caption_id in biases:
            self.data.append(
                (
                    caption_id,
                    captions[caption_id][0],
                    biases[caption_id]
                )
            )

        self.bias_classes_final = bias_classes_final
        self.captions = captions

    def __len__(self):
        return len(self.data)

    def get_data(self):
        return self.data

    def get_accuracy_info(self):
        return self.bias_classes_final, self.captions

    def __getitem__(self, index):
        # get image info
        caption_id, caption, proposed_biases = self.data[index]
        return caption_id, caption, proposed_biases