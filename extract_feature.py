import os
import torch
from PIL import Image
from torchvision import transforms

from deploy_fcn import DeployPredictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FeatureExtractor():
	def __init__(self, image_dir='feature/',
				feature_dir='./'):
		self.image_dir = image_dir.replace("'", "")
		self.feature_dir = feature_dir.replace("'", "")
		self.data_dir = 'models/'.replace("'", "")
		self.sample_dir = 'samples/'.replace("'", "")
		#self.model_list = next(os.walk(image_dir))[1]
		self.view_size = 48
		self.width = 227
		self.height = 227
		self.channel = 3
		self.transform = transforms.Compose([
            transforms.Resize([227, 227]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
		self.model = DeployPredictor()
		model_path = os.path.join('model/', 'lmvcnn_model.pt').replace("'", "")
		self.model.load_state_dict(torch.load(model_path))
		if not os.path.isdir(feature_dir):
			os.system('mkdir', feature_dir)
		self.batch_list = []

	def extractor(self, category, model):
		out = os.path.join(self.feature_dir, "feature.csv").replace("\\", "/")
		images = self.get_images(category, model)
		self.preprocess(images)
		
		# process the feature-extracting network
		with open(out, 'a') as f:
			for i, batch in enumerate(self.batch_list):
				# a feature for a sample point is stored in one line
				feature = self.model.forward(batch)
				f.write(model + "_" + str(2*i) + ",")
				f.write(category + ",")
				#print(feature[0])
				for value in feature[0]:
					f.write("%.4f " % value)
					f.write(",")
				f.write("\n")
				f.write(model + "_" + str(2*i+1) + ",")
				f.write(category + ",")
				for value in feature[1]:
					f.write("%.4f " % value)
					f.write(",")
				f.write("\n")

		self.save_sample(category, model)

	def single_extractor(self, obj_id, obj_path, images):
		"""
		extractor for a query object from the client
		or extract a feature vector for any objects from the given path
		"""
		#out = os.path.join(self.feature_dir, obj_id+'.csv').replace("\\", "/")
		batch = self.single_preprocess(images)

		# process the feature-extracting network
		return self.model.forward(batch)[0].detach().numpy()

	def get_images(self, category, model):
		# if no images found in the path, generate images
		img_path = os.path.join(self.image_dir, model).replace("\\", "/")
		#if not os.path.isdir(img_path) or not os.listdir(img_path):
		img_files = os.listdir(img_path)

		curr_idx = 0
		images = []
		point_images = []
		for img in img_files:
			#print(img)
			idx = img.find('_')
			point_idx = int(img[idx+1:idx+2])
			#print("point idx: " + str(point_idx))
			if point_idx == curr_idx:
				point_images.append(img)
			else:
				images.append(point_images)
				curr_idx = point_idx
				#print("point_images pre: " + str(len(point_images)))
				point_images = []
				#print(len(images[curr_idx-1]))
				point_images.append(img)

		images.append(point_images)

		return images

	def save_sample(self, category, model):
		out = os.path.join(self.feature_dir, "samples.csv").replace("\\", "/")
		sample = os.path.join(self.sample_dir, category, model + "_2.txt").replace("\\", "/")
		
		# store sample points to remember the position
		f_sample = open(sample, 'r')
		samples = f_sample.readlines()
		f_sample.close()

		with open(out, 'a') as out:
			for i, line in enumerate(samples):
				out.write(model + "_" + str(i) + ",")
				line = line.replace(" ", ",")
				out.write(line)

	def preprocess(self, images):
		self.batch_list.clear()
		zeros = torch.zeros(())
		batch = zeros.new_empty((self.height, self.width, self.channel, 2*self.view_size))
		for i, group in enumerate(images):
			if i % 2 == 0:
				batch = zeros.new_empty((self.height, self.width, self.channel, 2*self.view_size))
			for idx, img in enumerate(group):
				_idx = img.find('_')
				model_id = img[:_idx]
				img_path = os.path.join(self.image_dir, model_id).replace("\\", "/")
				img_path = os.path.join(img_path, img).replace("\\", "/")
				img = Image.open(img_path).convert('RGB')
				if self.transform is not None:
					img = self.transform(img)
					if idx % 2 == 0:
						batch[:, :, :, idx] = img.permute(2, 1, 0)
					else:
						batch[:, :, :, idx+48] = img.permute(2, 1, 0)
			if i % 2 == 1:
				self.batch_list.append(batch.permute(3, 2, 0, 1))

	def single_preprocess(self, images):
		zeros = torch.zeros(())
		batch = zeros.new_empty((self.height, self.width, self.channel, 2*self.view_size))
		for idx, img in enumerate(images):
			_idx = img.find('_')
			model_id = img[:_idx]
			img_path = os.path.join(self.image_dir, model_id).replace("\\", "/")
			img_path = os.path.join(img_path, img).replace("\\", "/")
			img = Image.open(img_path).convert('RGB')
			if self.transform is not None:
				img = self.transform(img)
				batch[:, :, :, idx] = img.permute(2, 1, 0)
				batch[:, :, :, idx+48] = img.permute(2, 1, 0)

		return batch.permute(3, 2, 0, 1)

if __name__ == '__main__':
	fe = FeatureExtractor()
