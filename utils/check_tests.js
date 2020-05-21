const fs = require('fs');
const xml2js = require('xml2js');

const TESTS_PATH = 'features/';

let allTests = {};

function readTestResults(callback) {
    fs.readdir(TESTS_PATH, function (err, testFolders) {
        if (err) console.log(err);
        testFolders.forEach(function (folderName) {
            const testPath = TESTS_PATH + '/' + folderName;
            if (fs.existsSync(testPath + '/test_results.xml')) {
                fs.readFile(testPath + '/test_results.xml', function (err, data) {
                    xml2js.parseString(data, function (err, xmlData) {
                        if (err) console.log(err);
                        allTests[testPath] = parseInt(xmlData['testsuites']['$']['failures']);
                        callback(testFolders.length);
                    });
                });
            } else {
                allTests[testPath] = 'NONE';
                callback(testFolders.length);
            }
        });
    });
}

readTestResults(function (foldersLength) {
    if (Object.keys(allTests).length === foldersLength) {
        let total = Object.keys(allTests).length;
        let passed = Object.keys(allTests).filter(testName => allTests[testName] === 0);
        let failed = Object.keys(allTests).filter(testName => allTests[testName] === 1);
        let noResult = Object.keys(allTests).filter(testName => allTests[testName] === 'NONE');
        
        console.log('//--------------------------');
        console.log('// PASSED');
        console.log('//--------------------------');
        passed.forEach(test => console.log('PASSED: ' + test.replace('features//', '')));
        console.log('//--------------------------');
        console.log('// FAILED');
        console.log('//--------------------------');
        failed.forEach(test => console.log('FAILED: ' + test.replace('features//', '')));
        console.log('//--------------------------');
        console.log('// NO RESULT (Compiler Errors)');
        console.log('//--------------------------');
        noResult.forEach(test => console.log('NO RESULT: ' + test.replace('features//', '')));
        

        console.log('\n\n');
        console.log('//--------------------------');
        console.log('// SUMMARY');
        console.log('//--------------------------');
        

        console.log('TOTAL: \t\t' + total + '\t100%');
        console.log('PASSED: \t' + passed.length + '\t' + (passed.length / total * 100) + '%');
        console.log('FAILED: \t' + failed.length + '\t' + (failed.length / total * 100) + '%');
        console.log('NO RESULT: \t' + noResult.length + '\t' + (noResult.length / total * 100) + '%');

    }
});