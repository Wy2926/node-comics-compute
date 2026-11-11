package main

import (
	"archive/zip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestUpdateVersionOrdering(t *testing.T) {
	for _, tt := range []struct {
		next, current string
		newer         bool
	}{
		{"v0.10.0", "0.9.9", true}, {"0.4.0", "0.4.0-dev", true},
		{"0.4.0", "0.4.0", false}, {"0.3.9", "0.4.0", false},
	} {
		got, err := newerRelease(tt.next, tt.current)
		if err != nil || got != tt.newer {
			t.Fatal(tt, got, err)
		}
	}
	for _, value := range []string{"", "v1", "1.2.3-rc1", "9999999999999.1.1", "01.2.3"} {
		if _, err := newerRelease(value, "0.3.0"); err == nil {
			t.Fatal("accepted", value)
		}
	}
}

func updateFixture(t *testing.T, a app) (string, releaseManifest, releaseManifest) {
	t.Helper()
	var old, next releaseManifest
	if err := readJSON(filepath.Join(a.root, "portable.json"), &old); err != nil {
		t.Fatal(err)
	}
	if err := readJSON(filepath.Join(a.root, "portable.json"), &next); err != nil {
		t.Fatal(err)
	}
	next.Version = "0.4.0"
	stage := filepath.Join(a.updateWork(), "next")
	for name := range old.Files {
		path := filepath.Join(stage, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(filepath.Join(a.root, filepath.FromSlash(name)))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	for name, content := range map[string]string{
		"Start.cmd": "new start", "Update.cmd": "new updater",
		"source/compute-node/update.ps1":                 "new helper",
		"source/classic-engine/classic_node/__main__.py": "new application",
		"source/compute-node/run_node.py":                "new runner",
	} {
		path := filepath.Join(stage, filepath.FromSlash(name))
		_ = os.MkdirAll(filepath.Dir(path), 0700)
		if err := os.WriteFile(path, []byte(content), 0600); err != nil {
			t.Fatal(err)
		}
		hash := sha256.Sum256([]byte(content))
		next.Files[name] = hex.EncodeToString(hash[:])
	}
	if err := writeJSON(filepath.Join(stage, "portable.json"), next); err != nil {
		t.Fatal(err)
	}
	return stage, old, next
}

func updateArchive(t *testing.T, stage string, manifest releaseManifest, extra map[string]string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "release.zip")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	archive := zip.NewWriter(file)
	names := []string{"portable.json"}
	for name := range manifest.Files {
		names = append(names, name)
	}
	for _, name := range names {
		data, err := os.ReadFile(filepath.Join(stage, filepath.FromSlash(name)))
		if err != nil {
			t.Fatal(err)
		}
		writer, err := archive.Create("NodeComicsNode/" + name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := writer.Write(data); err != nil {
			t.Fatal(err)
		}
	}
	for name, data := range extra {
		writer, err := archive.Create("NodeComicsNode/" + name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := io.WriteString(writer, data); err != nil {
			t.Fatal(err)
		}
	}
	if err := archive.Close(); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestUpdateArchiveValidation(t *testing.T) {
	a := fixtureBundle(t, "cuda")
	stage, _, next := updateFixture(t, a)
	archive := updateArchive(t, stage, next, nil)
	if _, err := unpackUpdate(archive, t.TempDir(), next.Version); err != nil {
		t.Fatal(err)
	}
	if _, err := unpackUpdate(archive, t.TempDir(), "0.5.0"); err == nil {
		t.Fatal("accepted wrong version")
	}
	for _, name := range []string{"../outside.txt", "data/node.json", ".setup/cache", "payload/model", "source/CON.txt", "source/x:stream", "source/x. /file", "NODE.EXE", "source/untracked.py"} {
		t.Run(name, func(t *testing.T) {
			archive := updateArchive(t, stage, next, map[string]string{name: "unexpected"})
			if _, err := unpackUpdate(archive, t.TempDir(), next.Version); err == nil {
				t.Fatal("accepted", name)
			}
		})
	}
	_ = os.WriteFile(filepath.Join(stage, "source/compute-node/run_node.py"), []byte("corrupt"), 0600)
	archive = updateArchive(t, stage, next, nil)
	if _, err := unpackUpdate(archive, t.TempDir(), next.Version); err == nil {
		t.Fatal("accepted corrupt source")
	}
}

func TestUpdateSwitchPreservesPrivateDataAndPayload(t *testing.T) {
	a := fixtureBundle(t, "cuda")
	stage, old, next := updateFixture(t, a)
	private := map[string][]byte{}
	for _, name := range []string{"data/node.json", "payload/runtime/python.exe", "payload/release.json"} {
		private[name], _ = os.ReadFile(filepath.Join(a.root, name))
	}
	_ = os.MkdirAll(filepath.Join(a.root, ".setup/downloads"), 0700)
	_ = os.WriteFile(filepath.Join(a.root, ".setup/downloads/keep"), []byte("cached"), 0600)
	if err := a.switchUpdate(stage, old, next, false); err != nil {
		t.Fatal(err)
	}
	for name, before := range private {
		after, err := os.ReadFile(filepath.Join(a.root, name))
		if err != nil || string(before) != string(after) {
			t.Fatal("private data changed", name, err)
		}
	}
	if a.updatePending() {
		t.Fatal("committed switch left pending journal")
	}
	if err := verifyFiles(a.root, next.Files); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(a.root, ".setup/downloads/keep")); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(a.updateWork(), "previous/node.exe")); err != nil {
		t.Fatal("missing backup", err)
	}
}

func TestUpdateRollsBackFailedSwitch(t *testing.T) {
	a := fixtureBundle(t, "cuda")
	stage, old, next := updateFixture(t, a)
	// This fails after the code directories have moved, at final validation.
	_ = os.Remove(filepath.Join(stage, "source/compute-node/run_node.py"))
	if err := a.switchUpdate(stage, old, next, false); err == nil {
		t.Fatal("accepted incomplete switch")
	}
	if err := verifyFiles(a.root, old.Files); err != nil {
		t.Fatal("did not restore original files", err)
	}
	if a.updatePending() {
		t.Fatal("rollback left journal")
	}
	if _, err := os.Stat(filepath.Join(a.root, "Update.cmd")); !os.IsNotExist(err) {
		t.Fatal("new file survived rollback")
	}
}

func TestInterruptedUpdateBlocksLaunchAndRecoversWithoutSource(t *testing.T) {
	a := fixtureBundle(t, "cuda")
	_, old, _ := updateFixture(t, a)
	previous := filepath.Join(a.updateWork(), "previous")
	_ = os.MkdirAll(previous, 0700)
	if err := os.Rename(filepath.Join(a.root, "source"), filepath.Join(previous, "source")); err != nil {
		t.Fatal(err)
	}
	_ = writeJSON(filepath.Join(a.updateWork(), "transaction.json"), updateTransaction{Existing: map[string]bool{"source": true, "Update.cmd": true}})
	if a.launch("", "cuda", true, false) == nil {
		t.Fatal("started during incomplete update")
	}
	if a.supervise() == nil {
		t.Fatal("host started during incomplete update")
	}
	// Exercise the actual double-click recovery entry, including a missing source
	// tree: the fallback cannot depend on source/compute-node/update.ps1.
	data, err := os.ReadFile("Update.cmd")
	if err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(filepath.Join(a.root, "Update.cmd"), data, 0600)
	_ = os.WriteFile(filepath.Join(previous, "Update.cmd"), data, 0600)
	_ = backupUpdateFile(filepath.Join(a.root, "node.exe"), filepath.Join(a.root, ".setup/update-host.exe"))
	cmd := exec.Command("cmd.exe", "/c", filepath.Join(a.root, "Update.cmd"))
	cmd.Env = append(os.Environ(), "NODE_TEST_FIXTURE=1")
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatal(err, string(output))
	}
	if a.updatePending() {
		t.Fatal("double-click did not recover")
	}
	if err := verifyFiles(a.root, old.Files); err != nil {
		t.Fatal(err)
	}
}

func TestUpdateCannotOverwriteUnmanagedRootFile(t *testing.T) {
	a := fixtureBundle(t, "cuda")
	stage, old, next := updateFixture(t, a)
	_ = os.WriteFile(filepath.Join(a.root, "Update.cmd"), []byte("operator script"), 0600)
	if a.switchUpdate(stage, old, next, false) == nil {
		t.Fatal("overwrote operator script")
	}
	data, _ := os.ReadFile(filepath.Join(a.root, "Update.cmd"))
	if string(data) != "operator script" {
		t.Fatal("lost operator script")
	}
}

func TestUpdateDownloadIsBoundedAndCancellable(t *testing.T) {
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/redirect" {
			http.Redirect(w, r, "http://example.invalid/release", 302)
			return
		}
		w.WriteHeader(200)
		_, _ = io.WriteString(w, strings.Repeat("x", 2048))
	}))
	defer server.Close()
	original := http.DefaultTransport
	http.DefaultTransport = server.Client().Transport
	t.Cleanup(func() { http.DefaultTransport = original })
	if err := fetchUpdate(context.Background(), server.URL, io.Discard, 4096, "fixture"); err != nil {
		t.Fatal(err)
	}
	if fetchUpdate(context.Background(), server.URL, io.Discard, 1024, "fixture") == nil {
		t.Fatal("ignored size bound")
	}
	if fetchUpdate(context.Background(), server.URL+"/redirect", io.Discard, 4096, "fixture") == nil {
		t.Fatal("accepted HTTP redirect")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if fetchUpdate(ctx, server.URL, io.Discard, 4096, "fixture") == nil {
		t.Fatal("ignored cancellation")
	}
}

func TestUpdateRejectsPrivateRecoveryPaths(t *testing.T) {
	a := fixtureBundle(t, "cuda")
	before, _ := os.ReadFile(a.configPath())
	_ = writeJSON(filepath.Join(a.updateWork(), "transaction.json"), updateTransaction{Existing: map[string]bool{"data": false}})
	if a.restoreUpdate() == nil {
		t.Fatal("accepted private recovery path")
	}
	after, _ := os.ReadFile(a.configPath())
	if string(before) != string(after) {
		t.Fatal("recovery deleted private identity")
	}
}

func TestUpdatePreservesCorruptRecoveryState(t *testing.T) {
	// Interrupted recovery must reject corrupt state instead of deleting backups.
	a := fixtureBundle(t, "cuda")
	_ = os.MkdirAll(a.updateWork(), 0700)
	_ = os.WriteFile(filepath.Join(a.updateWork(), "transaction.json"), []byte("{"), 0600)
	if a.restoreUpdate() == nil || !a.updatePending() {
		t.Fatal("ignored corrupt recovery state")
	}
}

func TestUpdateReplacesChangedEnvironmentAndRetainsBackup(t *testing.T) {
	a := fixtureBundle(t, "cuda")
	stage, old, next := updateFixture(t, a)
	prepared := fixtureBundle(t, "cuda")
	if err := os.Rename(prepared.bundleRoot(), filepath.Join(stage, "payload")); err != nil {
		t.Fatal(err)
	}
	data := []byte("changed-dependency==2\n")
	hash := sha256.Sum256(data)
	next.Requirements["cuda"] = hex.EncodeToString(hash[:])
	_ = os.WriteFile(filepath.Join(stage, "payload/runtime-requirements.txt"), data, 0600)
	_ = writeJSON(filepath.Join(stage, "portable.json"), next)
	if err := a.switchUpdate(stage, old, next, true); err != nil {
		t.Fatal(err)
	}
	if err := a.checkPayload("cuda"); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(filepath.Join(a.updateWork(), "previous/payload/runtime-requirements.txt"))
	if err != nil || string(before) != "fixture" {
		t.Fatal("lost old environment", err)
	}
}

func TestUpdateDrainCheckPreservesPendingDeliveryAndDocuments(t *testing.T) {
	t.Setenv("NODE_TEST_FIXTURE", "1")
	a := fixtureBundle(t, "cuda")
	if err := a.ensureUpdateDrained(); err != nil {
		t.Fatal(err)
	}
	_ = os.MkdirAll(a.statePath(), 0700)
	pending := filepath.Join(a.statePath(), "pending")
	_ = os.WriteFile(pending, []byte("frozen delivery"), 0600)
	if a.ensureUpdateDrained() == nil {
		t.Fatal("ignored pending delivery")
	}
	if data, _ := os.ReadFile(pending); string(data) != "frozen delivery" {
		t.Fatal("changed frozen result")
	}
	_ = os.Remove(pending)
	_ = os.MkdirAll(filepath.Join(a.statePath(), "documents/unfinished"), 0700)
	if a.ensureUpdateDrained() == nil {
		t.Fatal("ignored document workspace")
	}
}

func TestUpdateExcludesRunningHostsWithExternalData(t *testing.T) {
	a := fixtureBundle(t, "cuda")
	a.home = filepath.Join(t.TempDir(), "external-data")
	_ = writeJSON(a.configPath(), map[string]any{})
	launchFixture(t, a)
	lock, err := acquireLock(filepath.Join(a.root, ".setup/runtime.lock"))
	if err == nil {
		lock.Close()
		t.Fatal("allowed switching files under another running home")
	}
	if err := a.stop(); err != nil {
		t.Fatal(err)
	}
	lock, err = acquireLock(filepath.Join(a.root, ".setup/runtime.lock"))
	if err != nil {
		t.Fatal(err)
	}
	lock.Close()
}

type updateTransport func(*http.Request) (*http.Response, error)

func (f updateTransport) RoundTrip(request *http.Request) (*http.Response, error) { return f(request) }

func TestUpdateDownloadSwitchAndRestart(t *testing.T) {
	for _, corrupt := range []bool{false, true} {
		t.Run(map[bool]string{false: "success", true: "checksum-failure"}[corrupt], func(t *testing.T) {
			t.Setenv("NODE_TEST_FIXTURE", "1")
			a := fixtureBundle(t, "cuda")
			var current releaseManifest
			_ = readJSON(filepath.Join(a.root, "portable.json"), &current)
			current.Version = "0.3.0"
			_ = writeJSON(filepath.Join(a.root, "portable.json"), current)
			stage, old, next := updateFixture(t, a)
			archive := updateArchive(t, stage, next, nil)
			zipBytes, _ := os.ReadFile(archive)
			hash := sha256.Sum256(zipBytes)
			checksum := hex.EncodeToString(hash[:])
			if corrupt {
				checksum = strings.Repeat("0", 64)
			}
			metadata, _ := json.Marshal(map[string]any{"tag_name": "v0.4.0", "assets": []map[string]string{
				{"name": "NodeComicsNode.zip", "browser_download_url": releasePrefix + "v0.4.0/NodeComicsNode.zip"},
				{"name": "NodeComicsNode.zip.sha256", "browser_download_url": releasePrefix + "v0.4.0/NodeComicsNode.zip.sha256"},
			}})
			original := http.DefaultTransport
			http.DefaultTransport = updateTransport(func(request *http.Request) (*http.Response, error) {
				content := ""
				switch request.URL.String() {
				case releaseAPI:
					content = string(metadata)
				case releasePrefix + "v0.4.0/NodeComicsNode.zip":
					content = string(zipBytes)
				case releasePrefix + "v0.4.0/NodeComicsNode.zip.sha256":
					content = checksum + "  NodeComicsNode.zip\n"
				default:
					t.Fatal("unexpected request", request.URL)
				}
				return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(content)), ContentLength: int64(len(content))}, nil
			})
			t.Cleanup(func() { http.DefaultTransport = original })
			launchFixture(t, a)
			before, _ := a.liveHost()
			identity, _ := os.ReadFile(a.configPath())
			err := a.update()
			after, alive := a.liveHost()
			if corrupt {
				if err == nil || !alive || before.PID != after.PID {
					t.Fatal("download failure displaced current worker", err)
				}
				if err := verifyFiles(a.root, old.Files); err != nil {
					t.Fatal(err)
				}
			} else {
				if err != nil || !alive || before.PID == after.PID || a.status()["connected"] != true {
					t.Fatal("upgrade did not restart", err)
				}
				if err := verifyFiles(a.root, next.Files); err != nil {
					t.Fatal(err)
				}
			}
			preserved, _ := os.ReadFile(a.configPath())
			if string(identity) != string(preserved) {
				t.Fatal("upgrade changed node identity")
			}
		})
	}
}
