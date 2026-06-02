#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <Vision/Vision.h>

static NSNumber *GPSCoordinate(NSDictionary *gps, NSString *key) {
    id value = gps[key];
    if ([value isKindOfClass:[NSNumber class]]) {
        return value;
    }
    if ([value isKindOfClass:[NSString class]]) {
        return @([(NSString *)value doubleValue]);
    }
    return nil;
}

static NSDictionary *ProcessImage(NSString *path) {
    NSURL *url = [NSURL fileURLWithPath:path];
    CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);
    if (!source) {
        return @{@"image_path": path, @"lines": @[], @"error": @"could_not_load_image"};
    }
    CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, NULL);
    if (!image) {
        CFRelease(source);
        return @{@"image_path": path, @"lines": @[], @"error": @"could_not_load_image"};
    }

    NSDictionary *properties = CFBridgingRelease(CGImageSourceCopyPropertiesAtIndex(source, 0, NULL)) ?: @{};
    NSDictionary *gps = properties[(NSString *)kCGImagePropertyGPSDictionary] ?: @{};
    NSNumber *latitude = GPSCoordinate(gps, (NSString *)kCGImagePropertyGPSLatitude);
    NSNumber *longitude = GPSCoordinate(gps, (NSString *)kCGImagePropertyGPSLongitude);
    NSString *latitudeRef = gps[(NSString *)kCGImagePropertyGPSLatitudeRef];
    NSString *longitudeRef = gps[(NSString *)kCGImagePropertyGPSLongitudeRef];
    if ([[latitudeRef uppercaseString] isEqualToString:@"S"]) {
        latitude = @(-fabs([latitude doubleValue]));
    }
    if ([[longitudeRef uppercaseString] isEqualToString:@"W"]) {
        longitude = @(-fabs([longitude doubleValue]));
    }

    VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] init];
    request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
    request.usesLanguageCorrection = YES;
    request.recognitionLanguages = @[@"en-US"];

    NSNumber *orientationValue = properties[(NSString *)kCGImagePropertyOrientation] ?: @1;
    CGImagePropertyOrientation orientation = (CGImagePropertyOrientation)[orientationValue unsignedIntValue];
    VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:image orientation:orientation options:@{}];
    NSError *performError = nil;
    [handler performRequests:@[request] error:&performError];
    NSArray *observations = request.results ?: @[];

    NSMutableArray *lines = [NSMutableArray array];
    for (VNRecognizedTextObservation *observation in observations) {
        VNRecognizedText *candidate = [[observation topCandidates:1] firstObject];
        if (candidate) {
            [lines addObject:@{@"text": candidate.string, @"confidence": @(candidate.confidence)}];
        }
    }

    NSMutableDictionary *result = [@{@"image_path": path, @"lines": lines} mutableCopy];
    if (latitude) {
        result[@"latitude"] = latitude;
    }
    if (longitude) {
        result[@"longitude"] = longitude;
    }
    if (performError) {
        result[@"error"] = performError.localizedDescription;
    }
    CGImageRelease(image);
    CFRelease(source);
    return result;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        for (int index = 1; index < argc; index++) {
            NSString *path = [NSString stringWithUTF8String:argv[index]];
            NSData *json = [NSJSONSerialization dataWithJSONObject:ProcessImage(path) options:0 error:nil];
            printf("%s\n", [[[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding] UTF8String]);
        }
    }
    return 0;
}
