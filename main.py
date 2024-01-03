from http.server import HTTPServer, CGIHTTPRequestHandler
import cgi
from detect import predict_image
import time

class ImageHandler(CGIHTTPRequestHandler):
    def do_POST(self):
        content_type, _ = cgi.parse_header(self.headers.get('Content-type'))

        if content_type == 'multipart/form-data':
            form_data = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST'}
            )

            if 'image' in form_data and form_data['image'].file:
                image_data = form_data['image'].file.read()

                # Process the image data here (you can save it, analyze it, etc.)
                # For simplicity, let's just return a response with the image size.
                # response_message = f"Image received. Size: {len(image_data)} bytes"
                with open("image.jpg", 'wb') as f:
                    f.write(image_data)
                    time.sleep(1)
                response_message=predict_image()
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(response_message.encode('utf-8'))
            else:
                self.send_response(400)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write("Bad Request: 'image' field not found.".encode('utf-8'))
        else:
            self.send_response(400)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write("Bad Request: Unsupported Content Type.".encode('utf-8'))

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, ImageHandler)
    print(f"Server running on port {port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()

