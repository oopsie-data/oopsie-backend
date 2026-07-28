# Third-Party Notices

## InsightFace SCRFD decoder

`oopsie_backend/face_blur/detectors/scrfd.py` contains an adapted, detector-only
implementation of the output decoding and non-maximum suppression in
[`python-package/insightface/model_zoo/retinaface.py`](https://github.com/deepinsight/insightface/blob/v0.7/python-package/insightface/model_zoo/retinaface.py).

MIT License

Copyright (c) 2022 Jiankang Deng and Jia Guo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The separately downloaded InsightFace model weights are governed by the
[InsightFace model license](https://github.com/deepinsight/insightface#license)
and are not distributed in this repository.

## ORB-HD/deface and CenterFace

The optional face-blur dependencies include ORB-HD/deface v1.5.0, and the
`deface-centerface` adapter provisions the unmodified `centerface.onnx` file
bundled with that release. The native comparison runner calls deface's public
CenterFace and video-anonymization implementation directly. ORB-HD/deface and
its bundled CenterFace implementation/model are distributed under the
[MIT License](https://github.com/ORB-HD/deface/blob/v1.5.0/LICENSE).
